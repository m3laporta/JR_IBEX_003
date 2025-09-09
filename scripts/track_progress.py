#!/usr/bin/env python3
"""
JR_IBEX_003 Progress Tracking System
Track analysis progress and automatically save state
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
import pickle
import shutil
import logging

class ProgressTracker:
    """Automatic progress tracking for JR_IBEX_003 analysis"""
    
    def __init__(self, auto_save_interval=600):  # 10 minutes default
        self.project_root = Path(__file__).parent.parent.absolute()
        self.auto_save_interval = auto_save_interval
        self.auto_save_thread = None
        self.running = False
        self.tracked_variables = {}
        self.checkpoints = []
        
        # Set up directories
        self.backup_dir = self.project_root / "results" / "daily_outputs"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('ProgressTracker')
        
    def start_auto_save(self):
        """Start automatic saving thread"""
        if self.running:
            print("Auto-save already running")
            return
            
        self.running = True
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop)
        self.auto_save_thread.daemon = True
        self.auto_save_thread.start()
        
        print(f"🔄 Auto-save started (interval: {self.auto_save_interval/60:.1f} minutes)")
        
    def stop_auto_save(self):
        """Stop automatic saving"""
        self.running = False
        if self.auto_save_thread:
            self.auto_save_thread.join()
        print("⏹ Auto-save stopped")
        
    def _auto_save_loop(self):
        """Auto-save loop (runs in separate thread)"""
        while self.running:
            try:
                self._perform_auto_save()
                time.sleep(self.auto_save_interval)
            except Exception as e:
                self.logger.error(f"Auto-save error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
                
    def _perform_auto_save(self):
        """Perform automatic save"""
        timestamp = datetime.now()
        
        # Save tracked variables
        if self.tracked_variables:
            save_path = self.backup_dir / f"autosave_{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
            
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'timestamp': timestamp.isoformat(),
                        'variables': self.tracked_variables,
                        'checkpoints': self.checkpoints
                    }, f)
                
                self.logger.info(f"Auto-saved {len(self.tracked_variables)} variables")
                
                # Clean up old auto-saves (keep last 10)
                self._cleanup_old_saves()
                
            except Exception as e:
                self.logger.error(f"Failed to auto-save: {e}")
                
    def _cleanup_old_saves(self):
        """Clean up old auto-save files"""
        auto_save_files = sorted(self.backup_dir.glob("autosave_*.pkl"))
        
        if len(auto_save_files) > 10:
            for old_file in auto_save_files[:-10]:
                old_file.unlink()
                
    def track_variable(self, name, value, description=""):
        """Track a variable for auto-saving"""
        self.tracked_variables[name] = {
            'value': value,
            'description': description,
            'last_updated': datetime.now().isoformat(),
            'type': type(value).__name__
        }
        
        print(f"📌 Tracking variable: {name} ({type(value).__name__})")
        
    def untrack_variable(self, name):
        """Stop tracking a variable"""
        if name in self.tracked_variables:
            del self.tracked_variables[name]
            print(f"📍 Stopped tracking: {name}")
        
    def create_checkpoint(self, name, description="", save_globals=False):
        """Create a named checkpoint"""
        timestamp = datetime.now()
        
        checkpoint = {
            'name': name,
            'description': description,
            'timestamp': timestamp.isoformat(),
            'tracked_variables': self.tracked_variables.copy()
        }
        
        if save_globals:
            # Save selected global variables (be careful with memory)
            checkpoint['globals'] = self._get_safe_globals()
        
        self.checkpoints.append(checkpoint)
        
        # Save checkpoint to file
        checkpoint_path = self.backup_dir / f"checkpoint_{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            
            print(f"🏁 Checkpoint created: {name}")
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            self.logger.error(f"Checkpoint save failed: {e}")
            
        return checkpoint_path
        
    def _get_safe_globals(self):
        """Get safe global variables for checkpointing"""
        import __main__
        safe_globals = {}
        
        # Only include basic types and small objects
        for name, value in __main__.__dict__.items():
            if not name.startswith('_'):
                try:
                    # Check if object is reasonably small and serializable
                    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                        if isinstance(value, (list, dict, tuple)) and len(str(value)) < 10000:
                            safe_globals[name] = value
                        elif not isinstance(value, (list, dict, tuple)):
                            safe_globals[name] = value
                except:
                    pass
                    
        return safe_globals
        
    def load_checkpoint(self, checkpoint_path):
        """Load a saved checkpoint"""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            print(f"📂 Loaded checkpoint: {checkpoint['name']}")
            print(f"   Created: {checkpoint['timestamp']}")
            print(f"   Description: {checkpoint.get('description', 'No description')}")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None
            
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoint_files = sorted(self.backup_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            print("No checkpoints found")
            return []
            
        print(f"\n📋 Available Checkpoints ({len(checkpoint_files)}):")
        print("=" * 50)
        
        checkpoints = []
        for i, cp_file in enumerate(checkpoint_files, 1):
            try:
                with open(cp_file, 'r') as f:
                    cp_data = json.load(f)
                
                name = cp_data.get('name', 'unknown')
                timestamp = cp_data.get('timestamp', 'unknown')
                description = cp_data.get('description', '')
                
                print(f"{i:2d}. {name}")
                print(f"    {timestamp}")
                if description:
                    print(f"    {description}")
                print()
                
                checkpoints.append({
                    'file': cp_file,
                    'data': cp_data
                })
                
            except Exception as e:
                print(f"    ❌ Error reading {cp_file.name}: {e}")
                
        return checkpoints
        
    def update_workflow_progress(self, notebook_name, status, completion_percent=None):
        """Update progress for a specific workflow notebook"""
        from resume_session import load_daily_state, save_daily_state
        
        state = load_daily_state()
        
        if "workflow_progress" not in state:
            state["workflow_progress"] = {}
            
        workflow_info = {
            "status": status,
            "last_run": datetime.now().isoformat(),
            "completion_percent": completion_percent
        }
        
        if status == "completed":
            workflow_info["completed"] = True
            
        state["workflow_progress"][notebook_name] = workflow_info
        save_daily_state(state)
        
        status_icon = "✅" if status == "completed" else "🔄" if status == "running" else "⭕"
        print(f"{status_icon} {notebook_name}: {status}")
        
    def log_analysis_step(self, step_name, details=None, sample_id=None):
        """Log completion of an analysis step"""
        timestamp = datetime.now()
        
        log_entry = {
            'step': step_name,
            'timestamp': timestamp.isoformat(),
            'sample_id': sample_id,
            'details': details
        }
        
        # Append to daily log
        log_file = self.backup_dir / f"analysis_log_{timestamp.strftime('%Y%m%d')}.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                daily_log = json.load(f)
        else:
            daily_log = {'date': timestamp.date().isoformat(), 'entries': []}
            
        daily_log['entries'].append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(daily_log, f, indent=2)
            
        print(f"📝 Logged: {step_name}")
        
    def get_session_summary(self):
        """Get summary of current session"""
        summary = {
            'tracked_variables': len(self.tracked_variables),
            'checkpoints': len(self.checkpoints),
            'auto_save_running': self.running,
            'backup_directory': str(self.backup_dir)
        }
        
        # Count recent auto-saves
        today = datetime.now().strftime('%Y%m%d')
        recent_saves = len(list(self.backup_dir.glob(f"autosave_{today}_*.pkl")))
        summary['auto_saves_today'] = recent_saves
        
        return summary

# Global instance for easy access
progress_tracker = ProgressTracker()

# Convenience functions
def track(name, value, description=""):
    """Track a variable"""
    progress_tracker.track_variable(name, value, description)
    
def checkpoint(name, description=""):
    """Create a checkpoint"""
    return progress_tracker.create_checkpoint(name, description)
    
def start_tracking():
    """Start auto-save tracking"""
    progress_tracker.start_auto_save()
    
def stop_tracking():
    """Stop auto-save tracking"""
    progress_tracker.stop_auto_save()
    
def log_step(step_name, details=None, sample_id=None):
    """Log an analysis step"""
    progress_tracker.log_analysis_step(step_name, details, sample_id)

def update_progress(notebook, status, percent=None):
    """Update workflow progress"""
    progress_tracker.update_workflow_progress(notebook, status, percent)

if __name__ == "__main__":
    # Demo usage
    print("🔄 JR_IBEX_003 Progress Tracker")
    print("=" * 40)
    
    tracker = ProgressTracker()
    print(f"Backup directory: {tracker.backup_dir}")
    
    # Show available checkpoints
    tracker.list_checkpoints()
    
    # Show session summary
    summary = tracker.get_session_summary()
    print("\n📊 Session Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")