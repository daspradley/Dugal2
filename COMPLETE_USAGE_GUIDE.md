# Complete Guide: UniversalFileHandler + TemporaryUpdateHandler + SyncManager

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SyncManager                          │
│         (Orchestrates everything, handles syncing)          │
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼──────────┐
        │ FileHandler    │    │ UpdateHandler   │
        │ (READ-ONLY)    │    │ (WRITE-ONLY)    │
        │                │    │                 │
        │ - Search items │    │ - Update items  │
        │ - Get values   │    │ - Track changes │
        │ - Excel/Sheets │    │ - Temp files    │
        └────────────────┘    └─────────────────┘
                │                     │
        ┌───────▼─────────────────────▼───────┐
        │        Source Files                 │
        │  - Excel (OneDrive/Local)           │
        │  - Google Sheets                    │
        │  - CSV                              │
        └─────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation
```bash
# Required for all features
pip install openpyxl --break-system-packages

# Optional: For Google Sheets support
pip install google-auth google-api-python-client --break-system-packages
```

### Basic Usage
```python
from sync_manager import SyncManager

# Initialize
sync = SyncManager(search_engine=my_search_engine)

# Open file (auto-detects type)
result = sync.open_file("C:/Users/.../OneDrive/Bar Inventory.xlsx")

if result['success']:
    print(f"✅ Opened: {result['file_info']['filename']}")
    print(f"📂 Temp: {result['temp_path']}")
    
    # Make updates (go to temp file)
    sync.update_inventory("Absolut", 2)
    sync.update_inventory("Titos", 5, is_addition=True)
    
    # Save back to source
    sync.save_to_source()
    
    # Close
    sync.close_file()
```

---

## 📖 Complete Examples

### Example 1: Basic Workflow
```python
from sync_manager import SyncManager

# Create manager
sync = SyncManager()

# Open file
sync.open_file("Bar Inventory.xlsx")

# Update inventory
sync.update_inventory("Belvedere", 3)
sync.update_inventory("Hendricks", -1, is_addition=True)

# Check status
status = sync.get_status()
print(f"Changes: {status['change_summary']['total_changes']}")

# Save
sync.save_to_source()  # Creates backup automatically

# Close
sync.close_file()
```

### Example 2: Google Sheets
```python
from sync_manager import SyncManager

# Initialize with Google credentials
sync = SyncManager(google_credentials_path="google_creds.json")

# Open Google Sheet
sync.open_file("https://docs.google.com/spreadsheets/d/ABC123/edit")

# Make updates (writes to local temp Excel file)
sync.update_inventory("Absolut", 5)

# Check changes
if sync.update_handler.has_unsaved_changes():
    changes = sync.update_handler.get_change_log()
    for change in changes:
        print(f"{change['item']}: {change['old_value']} → {change['new_value']}")

# Save back (uploads to Google Sheets - Phase 2 feature)
# sync.save_to_source()  # Coming soon!

# For now, save temp file manually:
temp_path = sync.update_handler.get_temp_path()
print(f"Download temp file from: {temp_path}")

sync.close_file()
```

### Example 3: Batch Updates
```python
from sync_manager import SyncManager

sync = SyncManager()
sync.open_file("inventory.xlsx")

# Prepare batch
updates = [
    {"item_name": "Absolut", "value": 10},
    {"item_name": "Titos", "value": 5, "is_addition": True},
    {"item_name": "Belvedere", "value": -2, "is_addition": True},
    {"item_name": "Hendricks", "value": 0}  # Set to zero
]

# Execute batch
result = sync.batch_update(updates)
print(f"✅ {result['successful']} succeeded")
print(f"❌ {result['failed']} failed")

# Save all changes
sync.save_to_source()
sync.close_file()
```

### Example 4: Change Tracking
```python
from sync_manager import SyncManager

sync = SyncManager()
sync.open_file("inventory.xlsx")

# Make several updates
sync.update_inventory("Absolut", 5)
sync.update_inventory("Titos", 3, is_addition=True)
sync.update_inventory("Belvedere", 0)

# Get change summary
summary = sync.update_handler.get_change_summary()
print(f"Total changes: {summary['total_changes']}")
print(f"Items modified: {summary['items_modified']}")
print(f"First change: {summary['first_change']}")
print(f"Last change: {summary['last_change']}")

# Get detailed log
log = sync.update_handler.get_change_log()
for entry in log:
    print(f"{entry['timestamp']}: {entry['item']} - "
          f"{entry['old_value']} → {entry['new_value']} ({entry['operation']})")

# Export log
sync.update_handler.export_change_log("changes_today.json")

sync.save_to_source()
sync.close_file()
```

### Example 5: Discard Changes
```python
from sync_manager import SyncManager

sync = SyncManager()
sync.open_file("inventory.xlsx")

# Make some updates
sync.update_inventory("Absolut", 100)
sync.update_inventory("Titos", 200)

# Oops, made a mistake!
print("Discarding changes...")
sync.discard_changes()  # Recreates temp from source

# Start fresh
sync.update_inventory("Absolut", 5)  # Correct value
sync.save_to_source()
sync.close_file()
```

### Example 6: Conflict Detection
```python
from sync_manager import SyncManager

sync = SyncManager()
sync.open_file("shared_inventory.xlsx")

# Make changes
sync.update_inventory("Absolut", 5)

# Meanwhile, another user edits the OneDrive file...

# Check for conflicts
conflict = sync.check_for_conflicts()
if conflict['has_conflict']:
    print("⚠️ WARNING: Someone else modified the file!")
    print("Options:")
    print("1. Save anyway (overwrites their changes)")
    print("2. Discard your changes and reload theirs")
    print("3. Export your changes to review")
    
    # Option 2: Reload their changes
    sync.reload_from_source()
else:
    # No conflict, safe to save
    sync.save_to_source()

sync.close_file()
```

### Example 7: Context Manager (Clean Cleanup)
```python
from sync_manager import SyncManager

# Automatic cleanup
with SyncManager() as sync:
    sync.open_file("inventory.xlsx")
    sync.update_inventory("Absolut", 5)
    sync.save_to_source()
    # File automatically closed when leaving 'with' block
```

---

## 🔥 Integration with Voice System

### Replace Old Update Logic
```python
# OLD CODE (in voice_interaction.py)
def _execute_inventory_update(self, item_name, value, is_addition):
    onedrive_handler = self.state.dugal.onedrive_handler
    update_result = onedrive_handler.update_inventory(item, value, is_addition)
    # ❌ File path issues, locking problems

# NEW CODE (in voice_interaction.py)
def _execute_inventory_update(self, item_name, value, is_addition):
    # Use SyncManager instead!
    result = self.sync_manager.update_inventory(item_name, value, is_addition)
    
    if result['success']:
        self.speak(f"Updated {item_name} to {result['new_value']}")
    else:
        self.speak(f"Error: {result['message']}")
    
    return result
```

### Initialize in Voice System
```python
# In voice_interaction.py __init__
from sync_manager import SyncManager

self.sync_manager = SyncManager(search_engine=self.search_engine)
```

### File Opening Hook
```python
# When user opens file via GUI
def on_file_opened(self, file_path):
    result = self.sync_manager.open_file(file_path)
    
    if result['success']:
        self.speak("File opened and ready for updates")
        print(f"Temp file: {result['temp_path']}")
    else:
        self.speak(f"Error opening file: {result['message']}")
```

### Periodic Auto-Save
```python
# In your main loop or timer
def auto_save_timer(self):
    if self.sync_manager.is_open:
        status = self.sync_manager.get_status()
        if status['has_unsaved_changes']:
            print("Auto-saving changes...")
            result = self.sync_manager.save_to_source()
            if result['success']:
                print(f"✅ Saved {result['changes_saved']} changes")
```

---

## 🎯 Key Methods Reference

### SyncManager
```python
# File operations
sync.open_file(path_or_url)          # Open any file type
sync.close_file(save_changes=False)  # Close file
sync.get_status()                    # Get current status

# Updates
sync.update_inventory(item, value, is_addition=False)
sync.batch_update(updates_list)

# Syncing
sync.save_to_source(backup=True)     # Save temp → source
sync.discard_changes()               # Reset temp file
sync.reload_from_source()            # Pull latest from source
sync.check_for_conflicts()           # Detect conflicts
```

### TemporaryUpdateHandler (via sync.update_handler)
```python
# Access directly for advanced features
sync.update_handler.has_unsaved_changes()
sync.update_handler.get_change_log()
sync.update_handler.get_change_summary()
sync.update_handler.export_change_log(path)
sync.update_handler.get_temp_path()
```

### UniversalFileHandler (via sync.file_handler)
```python
# Access directly for read operations
sync.file_handler.search_item(item_name)
sync.file_handler.get_item_value(item_name)
sync.file_handler.get_item_location(item_name)
sync.file_handler.get_sheet_names()
sync.file_handler.get_file_info()
```

---

## 🛡️ Best Practices

### 1. Always Use Context Manager
```python
with SyncManager() as sync:
    sync.open_file(path)
    # ... do work ...
    sync.save_to_source()
# Automatic cleanup
```

### 2. Check Status Before Saving
```python
status = sync.get_status()
if status['has_unsaved_changes']:
    conflict = sync.check_for_conflicts()
    if not conflict['has_conflict']:
        sync.save_to_source()
```

### 3. Export Change Logs for Auditing
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sync.update_handler.export_change_log(f"changes_{timestamp}.json")
```

### 4. Handle Errors Gracefully
```python
result = sync.open_file(path)
if not result['success']:
    logger.error(result['message'])
    # Fallback logic
```

---

## 🎉 What This Solves

✅ **No more file path confusion** - SyncManager knows where everything is  
✅ **No more locking issues** - Temp files are always writable  
✅ **Instant updates** - Writes to local disk are fast  
✅ **Multi-format support** - Excel, Google Sheets, CSV  
✅ **Change tracking** - Full audit log of every change  
✅ **Conflict detection** - Warns if source changed  
✅ **Undo capability** - Discard and start over anytime  
✅ **Clean architecture** - Each component has ONE job  

---

## 🚀 Next Steps

1. **Test with your existing files**
2. **Integrate with voice_interaction.py**
3. **Add GUI buttons** (Save, Discard, Status)
4. **Phase 2**: Google Sheets write-back support
5. **Phase 2**: Auto-sync timers

Let's replace that old tangled mess with this beautiful new architecture! 💪
