# UI Layout Optimization Plan - Final

## User Requirements (Confirmed)

### 1. Detection Results Tab (`results_tab.py`)
**Goal**: Optimize frame preview layout
- ✅ Display 4 frames horizontally (already implemented: `cols = 4`)
- ✅ Keep current frame size (150x150) - no changes needed
- Ensure no horizontal scrollbar
- Ensure table text is not obscured (already fixed in previous session)

**Implementation**:
- Add `setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)` to frame preview scroll area
- Keep existing 4-column grid layout
- Keep existing frame image size (150x150)

**Files to modify**:
- `src/gui/results_tab.py`
  - Line ~174: Add horizontal scroll bar policy

---

### 2. Model Management Tab (`model_management_tab.py`)

#### Change 2.1: Optimize Action Buttons Layout
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 168-186 (action buttons section)

**User Requirements**:
- Button height: **50px** (standard button height)
- Remove width constraints (allow buttons to expand horizontally)
- Buttons should fill horizontal space in their section

**Current Code**:
```python
action_layout = QHBoxLayout()
action_layout.setSpacing(8)

self.delete_button = QPushButton("删除")
self.delete_button.setEnabled(False)
self.delete_button.setMaximumWidth(100)  # REMOVE THIS
self.delete_button.clicked.connect(self.delete_model)
self.delete_button.setStyleSheet(danger_button_style)
action_layout.addWidget(self.delete_button)

self.set_default_button = QPushButton("设为默认")
self.set_default_button.setEnabled(False)
self.set_default_button.setMaximumWidth(110)  # REMOVE THIS
self.set_default_button.clicked.connect(self.set_as_default)
self.set_default_button.setStyleSheet(secondary_button_style)
action_layout.addWidget(self.set_default_button)

left_layout.addLayout(action_layout)
```

**New Code**:
```python
action_layout = QHBoxLayout()
action_layout.setSpacing(8)

self.delete_button = QPushButton("删除")
self.delete_button.setEnabled(False)
self.delete_button.setMinimumHeight(50)  # ADD: Set minimum height
# REMOVED: setMaximumWidth constraint
self.delete_button.clicked.connect(self.delete_model)
self.delete_button.setStyleSheet(danger_button_style)
action_layout.addWidget(self.delete_button)

self.set_default_button = QPushButton("设为默认")
self.set_default_button.setEnabled(False)
self.set_default_button.setMinimumHeight(50)  # ADD: Set minimum height
# REMOVED: setMaximumWidth constraint
self.set_default_button.clicked.connect(self.set_as_default)
self.set_default_button.setStyleSheet(secondary_button_style)
action_layout.addWidget(self.set_default_button)

action_layout.addStretch()  # ADD: Allow horizontal expansion
left_layout.addLayout(action_layout)
```

#### Change 2.2: Optimize TensorBoard Log Button
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 314-315 (TensorBoard button creation)

**Current Code**:
```python
self.select_log_button = QPushButton("选择TensorBoard日志 Select Log")
self.select_log_button.setMaximumWidth(200)  # REMOVE THIS
self.select_log_button.clicked.connect(self.select_tensorboard_log)
```

**New Code**:
```python
self.select_log_button = QPushButton("选择TensorBoard日志 Select Log")
self.select_log_button.setMinimumHeight(50)  # ADD: Match action button height
# REMOVED: setMaximumWidth constraint
self.select_log_button.clicked.connect(self.select_tensorboard_log)
```

#### Change 2.3: Simplify Model Details (Remove Modified Time)
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 241-246 (detail labels creation)

**Current Code**:
```python
# Create detail labels - compact format
self.filename_label = QLabel("文件: -")
self.path_label = QLabel("路径: -")
self.size_label = QLabel("大小: -")
self.modified_label = QLabel("修改: -")  # REMOVE THIS LINE

# Add separator
separator = QLabel()
separator.setFrameStyle(QLabel.HLine | QLabel.Sunken)
details_layout.addWidget(separator, 4, 0, 1, 2)

# Model architecture details - compact
self.backbone_label = QLabel("骨干: -")
...
```

**New Code**:
```python
# Create detail labels - compact format
self.filename_label = QLabel("文件: -")
self.path_label = QLabel("路径: -")
self.size_label = QLabel("大小: -")
# REMOVED: self.modified_label = QLabel("修改: -")

# Add separator
separator = QLabel()
separator.setFrameStyle(QLabel.HLine | QLabel.Sunken)
details_layout.addWidget(separator, 3, 0, 1, 2)  # Row index reduced by 1

# Model architecture details - compact
self.backbone_label = QLabel("骨干: -")
...
```

**Note**: Must update all subsequent `details_layout.addWidget()` calls since row indices shift after removing one label

---

## Files to Modify Summary

### 1. `src/gui/results_tab.py`
- **Line ~174**: Add `setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)` to frame preview scroll area

### 2. `src/gui/model_management_tab.py`
- **Line ~175**: Remove `setMaximumWidth(100)`, add `setMinimumHeight(50)`
- **Line ~181**: Remove `setMaximumWidth(110)`, add `setMinimumHeight(50)`
- **Line ~186**: Add `action_layout.addStretch()`
- **Line ~315**: Remove `setMaximumWidth(200)`, add `setMinimumHeight(50)`
- **Line ~244**: Remove `self.modified_label` creation
- **Lines ~246-276**: Update row indices in `details_layout.addWidget()` calls (subtract 1 from each)

---

## Expected Outcome

### Detection Results Tab
- ✅ Frame preview displays 4 frames horizontally
- ✅ No horizontal scrollbar needed
- ✅ Table text remains fully visible (already fixed)
- ✅ Maintains current frame image size (150x150)

### Model Management Tab
- ✅ Action buttons (delete, set default) are 50px tall
- ✅ Buttons expand horizontally to fill available space
- ✅ TensorBoard log button matches action button height (50px)
- ✅ Model details panel uses less vertical space (10 labels instead of 11)
- ✅ Overall more balanced layout with better vertical space utilization

---

## Verification Steps

### Test Detection Results Tab
1. Start application: `python scripts/app.py`
2. Navigate to "Detection Results" tab
3. Verify frame preview shows 4 frames in one row
4. Verify no horizontal scrollbar appears in frame preview area
5. Verify table headers are fully visible

### Test Model Management Tab
1. Navigate to "Model Management" tab
2. Verify delete button height is 50px
3. Verify set default button height is 50px
4. Verify buttons expand horizontally
5. Verify TensorBoard log button height is 50px
6. Verify "修改" (modified time) label is removed from model details
7. Verify model list loads correctly with color-coded categories

---

## Implementation Plan

### Phase 1: Detection Results Tab

#### Change 1.1: Optimize Frame Preview Grid Layout
**File**: `src/gui/results_tab.py`
**Location**: Lines 356-364 (frame display loop)

**Current Code**:
```python
# Display frames in a grid (4 columns for larger frame display)
cols = 4
for idx, frame_info in enumerate(frames):
    row_idx = idx // cols
    col_idx = idx % cols
    frame_widget = self.create_frame_preview(frame_info)
    self.frame_preview_layout.addWidget(frame_widget, row_idx, col_idx)
```

**Issue**: None - already uses 4 columns as requested

**Optimization**: Ensure scroll area is configured properly
- Add `setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)` to scroll area
- Ensure frame preview widget has proper size policy

#### Change 1.2: Adjust Frame Card Size
**File**: `src/gui/results_tab.py`
**Location**: Lines 366-396 (create_frame_preview method)

**Current Code**:
```python
scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
```

**Optimization**:
- Consider reducing frame image size from 150x150 to 140x140 or 130x130
- This allows 4 frames to fit better without horizontal scrolling

**Files to modify**:
- `src/gui/results_tab.py`
  - Line ~175: Scroll area horizontal scroll policy
  - Line ~385: Frame image size (optional optimization)

---

### Phase 2: Model Management Tab

#### Change 2.1: Optimize Action Buttons Layout
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 168-186 (action buttons section)

**Current Code**:
```python
# Action buttons for selected model
action_layout = QHBoxLayout()
action_layout.setSpacing(8)

self.delete_button = QPushButton("删除")
self.delete_button.setEnabled(False)
self.delete_button.setMaximumWidth(100)  # Current fix
self.delete_button.clicked.connect(self.delete_model)
self.delete_button.setStyleSheet(danger_button_style)
action_layout.addWidget(self.delete_button)

self.set_default_button = QPushButton("设为默认")
self.set_default_button.setEnabled(False)
self.set_default_button.setMaximumWidth(110)  # Current fix
self.set_default_button.clicked.connect(self.set_as_default)
self.set_default_button.setStyleSheet(secondary_button_style)
action_layout.addWidget(self.set_default_button)

left_layout.addLayout(action_layout)
```

**Optimization**:
1. Remove maximum width constraints
2. Set minimum height larger (e.g., 50px instead of default)
3. Add stretch to allow horizontal expansion

**New Code**:
```python
# Action buttons for selected model
action_layout = QHBoxLayout()
action_layout.setSpacing(8)

self.delete_button = QPushButton("删除")
self.delete_button.setEnabled(False)
self.delete_button.setMinimumHeight(50)  # Taller button
self.delete_button.clicked.connect(self.delete_model)
self.delete_button.setStyleSheet(danger_button_style)
action_layout.addWidget(self.delete_button)

self.set_default_button = QPushButton("设为默认")
self.set_default_button.setEnabled(False)
self.set_default_button.setMinimumHeight(50)  # Taller button
self.set_default_button.clicked.connect(self.set_as_default)
self.set_default_button.setStyleSheet(secondary_button_style)
action_layout.addWidget(self.set_default_button)

action_layout.addStretch()  # Allow horizontal expansion
left_layout.addLayout(action_layout)
```

#### Change 2.2: Optimize TensorBoard Log Button
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 314-315 (in curves_tab creation)

**Current Code**:
```python
self.select_log_button = QPushButton("选择TensorBoard日志 Select Log")
self.select_log_button.setMaximumWidth(200)  # Current fix
self.select_log_button.clicked.connect(self.select_tensorboard_log)
```

**Optimization**:
1. Remove maximum width constraint
2. Set minimum height to match action buttons (50px)
3. Allow button to expand horizontally

**New Code**:
```python
self.select_log_button = QPushButton("选择TensorBoard日志 Select Log")
self.select_log_button.setMinimumHeight(50)  # Match action buttons
self.select_log_button.clicked.connect(self.select_tensorboard_log)
```

#### Change 2.3: Simplify Model Details (Reduce Vertical Space)
**File**: `src/gui/model_management_tab.py`
**Location**: Lines 214-280 (model details section)

**Current Labels** (11 total):
1. 文件
2. 路径
3. 大小
4. 修改 - **CAN REMOVE**
5. 骨干
6. 数据集
7. 段数
8. 每段帧数
9. 总帧数
10. 轮数
11. 精度
12. 状态

**Optimization**: Remove less critical labels
- Remove: 修改时间
- Consider removing: 总帧数 (can be calculated)
- Keep essential info only

**Result**: 2-3 rows saved, reducing panel height

---

## Files to Modify

### 1. `src/gui/results_tab.py`
**Changes**:
1. Line ~175: Add `setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)` to frame preview scroll area
2. Line ~385: Optionally reduce frame image size from 150 to 140

### 2. `src/gui/model_management_tab.py`
**Changes**:
1. Line ~175: Remove `delete_button.setMaximumWidth(100)`, add `setMinimumHeight(50)`
2. Line ~182: Remove `set_default_button.setMaximumWidth(110)`, add `setMinimumHeight(50)`
3. Line ~186: Add `action_layout.addStretch()` for horizontal expansion
4. Line ~315: Remove `select_log_button.setMaximumWidth(200)`, add `setMinimumHeight(50)`
5. Lines ~244-246: Remove "修改" (modified time) label

---

## Summary

### Detection Results Tab
- ✅ Already uses 4-column grid layout
- ✅ Add horizontal scroll bar disable
- ✅ Optional: Reduce frame size slightly

### Model Management Tab
- ✅ Make buttons taller (50px min height)
- ✅ Remove width constraints
- ✅ Add stretch for horizontal expansion
- ✅ Remove less critical model details (modified time)

### Expected Outcome
- Frame preview shows 4 frames without horizontal scrolling
- Action buttons are more prominent (taller and wider)
- Model management panel uses less vertical space
- Overall cleaner, more balanced layout
