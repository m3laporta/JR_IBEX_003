#!/usr/bin/env python3
"""
JR_IBEX_003 Session Resume System
Resume exactly where you left off from previous session
"""

import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging

def get_daily_state_path():
    """Get path to daily state file"""
    project_root = Path(__file__).parent.parent.absolute()
    return project_root / "config" / "daily_state.json"

def get_template_state_path():
    """Get path to template state file"""
    project_root = Path(__file__).parent.parent.absolute()
    return project_root / "config" / "daily_state_template.json"

def load_daily_state():
    """Load current daily state or create from template"""
    state_path = get_daily_state_path()
    template_path = get_template_state_path()
    
    if not state_path.exists():
        if template_path.exists():
            print("📄 Creating new daily state from template...")
            shutil.copy(template_path, state_path)
        else:
            print("⚠ No state template found, creating minimal state")
            create_minimal_state(state_path)
    
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        return state
    except json.JSONDecodeError as e:
        print(f"❌ Error loading daily state: {e}")
        return create_minimal_state(state_path)

def create_minimal_state(state_path):
    """Create minimal state structure"""
    minimal_state = {
        "session_info": {
            "last_session_date": None,
            "session_start_time": None,
            "user": "mlaporte",
            "hostname": "implk-01"
        },
        "current_sample": {
            "roi_id": None,
            "current_workflow_step": "startup"
        },
        "workflow_progress": {},
        "notes": {
            "daily_goals": [],
            "completed_tasks": [],
            "next_session_plan": []
        }
    }
    
    with open(state_path, 'w') as f:
        json.dump(minimal_state, f, indent=2)
    
    return minimal_state

def update_session_start(state):
    """Update session start information"""
    now = datetime.now()
    
    state["session_info"]["session_start_time"] = now.isoformat()
    state["session_info"]["last_session_date"] = now.date().isoformat()
    
    # Log session continuation vs new session
    last_date = state["session_info"].get("last_session_date")
    if last_date and last_date != now.date().isoformat():
        print(f"🌅 Starting new day - Previous session: {last_date}")
    else:
        print(f"🔄 Continuing session from earlier today")
    
    return state

def save_daily_state(state):
    """Save current daily state"""
    state_path = get_daily_state_path()
    
    # Update session end time
    state["session_info"]["session_end_time"] = datetime.now().isoformat()
    
    try:
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Daily state saved to {state_path}")
    except Exception as e:
        print(f"❌ Error saving daily state: {e}")

def display_session_summary(state):
    """Display summary of current session state"""
    print("\n📊 Session Summary")
    print("=" * 40)
    
    # Session info
    session_info = state.get("session_info", {})
    print(f"User: {session_info.get('user', 'unknown')}")
    print(f"Host: {session_info.get('hostname', 'unknown')}")
    
    last_session = session_info.get("last_session_date")
    if last_session:
        print(f"Last session: {last_session}")
    
    # Current sample
    current_sample = state.get("current_sample", {})
    roi_id = current_sample.get("roi_id")
    if roi_id:
        print(f"Current sample: {roi_id}")
        workflow_step = current_sample.get("current_workflow_step", "unknown")
        print(f"Workflow step: {workflow_step}")
    else:
        print("No current sample set")
    
    # Workflow progress
    workflow_progress = state.get("workflow_progress", {})
    if workflow_progress:
        print("\nWorkflow Progress:")
        for step, info in workflow_progress.items():
            status = info.get("status", "unknown")
            completed = info.get("completed", False)
            status_icon = "✅" if completed else "⏳" if status == "running" else "⭕"
            print(f"  {status_icon} {step}: {status}")
    
    # Daily goals and tasks
    notes = state.get("notes", {})
    daily_goals = notes.get("daily_goals", [])
    completed_tasks = notes.get("completed_tasks", [])
    
    if daily_goals:
        print(f"\nDaily Goals ({len(daily_goals)}):")
        for i, goal in enumerate(daily_goals, 1):
            print(f"  {i}. {goal}")
    
    if completed_tasks:
        print(f"\nCompleted Today ({len(completed_tasks)}):")
        for task in completed_tasks[-5:]:  # Show last 5
            print(f"  ✓ {task}")

def check_for_auto_saves(state):
    """Check for and report auto-save files"""
    project_root = Path(__file__).parent.parent.absolute()
    backup_dir = project_root / "results" / "daily_outputs"
    
    if backup_dir.exists():
        today = datetime.now().date().isoformat()
        auto_saves = list(backup_dir.glob(f"*{today}*autosave*"))
        
        if auto_saves:
            print(f"\n💾 Found {len(auto_saves)} auto-save files from today:")
            for save_file in sorted(auto_saves)[-3:]:  # Show 3 most recent
                print(f"  📄 {save_file.name}")

def suggest_next_steps(state):
    """Suggest next steps based on current state"""
    print("\n🎯 Suggested Next Steps:")
    
    current_sample = state.get("current_sample", {})
    workflow_step = current_sample.get("current_workflow_step", "startup")
    
    if workflow_step == "startup":
        print("  1. Run 00_daily_startup.ipynb to set up today's analysis")
        print("  2. Load or select a sample for processing")
        
    elif workflow_step == "alignment":
        print("  1. Continue with spatial processing pipeline")
        print("  2. Check alignment quality and results")
        
    elif workflow_step == "spatial_processing":
        print("  1. Run advanced spatial analysis methods")
        print("  2. Apply literature-based analysis pipelines")
        
    else:
        print("  1. Continue from where you left off")
        print("  2. Check workflow progress in notebooks")
    
    # Check for incomplete goals
    notes = state.get("notes", {})
    daily_goals = notes.get("daily_goals", [])
    if daily_goals:
        print(f"  3. Work on remaining daily goals ({len(daily_goals)} pending)")

def set_sample_context(state, roi_id=None):
    """Set the current sample context"""
    if roi_id is None:
        # Try to detect from file system or user input
        project_root = Path(__file__).parent.parent.absolute()
        aligned_dir = project_root / "aligned"
        
        if aligned_dir.exists():
            sample_dirs = [d.name for d in aligned_dir.iterdir() if d.is_dir()]
            if sample_dirs:
                print(f"\nAvailable samples: {sample_dirs}")
                roi_id = input("Enter sample ID (or press Enter for no change): ").strip()
    
    if roi_id:
        state["current_sample"]["roi_id"] = roi_id
        print(f"🎯 Set current sample to: {roi_id}")
    
    return state

def main(roi_id=None, auto_continue=True):
    """Main resume session function"""
    print("🔄 JR_IBEX_003 Session Resume System")
    print("=" * 50)
    
    # Load current state
    print("📂 Loading daily state...")
    state = load_daily_state()
    
    # Update session start
    state = update_session_start(state)
    
    # Set sample context
    state = set_sample_context(state, roi_id)
    
    # Display summary
    display_session_summary(state)
    
    # Check for auto-saves
    check_for_auto_saves(state)
    
    # Suggest next steps
    suggest_next_steps(state)
    
    # Save updated state
    save_daily_state(state)
    
    print(f"\n✅ Session resumed successfully!")
    
    if auto_continue:
        print("\n🚀 Loading workspace...")
        # Import and run workspace loader
        from load_workspace import main as load_workspace
        workspace = load_workspace()
        
        return {
            'state': state,
            'workspace': workspace
        }
    
    return state

def add_daily_goal(goal_text):
    """Add a goal for today's session"""
    state = load_daily_state()
    
    if "notes" not in state:
        state["notes"] = {}
    if "daily_goals" not in state["notes"]:
        state["notes"]["daily_goals"] = []
    
    state["notes"]["daily_goals"].append(goal_text)
    save_daily_state(state)
    
    print(f"🎯 Added daily goal: {goal_text}")

def complete_task(task_text):
    """Mark a task as completed"""
    state = load_daily_state()
    
    if "notes" not in state:
        state["notes"] = {}
    if "completed_tasks" not in state["notes"]:
        state["notes"]["completed_tasks"] = []
    
    timestamp = datetime.now().strftime("%H:%M")
    completed_task = f"[{timestamp}] {task_text}"
    state["notes"]["completed_tasks"].append(completed_task)
    
    # Remove from daily goals if it exists
    daily_goals = state["notes"].get("daily_goals", [])
    state["notes"]["daily_goals"] = [g for g in daily_goals if task_text not in g]
    
    save_daily_state(state)
    print(f"✅ Completed: {task_text}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        roi_id = sys.argv[1]
        result = main(roi_id)
    else:
        result = main()