# Bazaar Tracker GUI Implementation

## Overview
A custom GUI for your Bazaar Price Prediction mod with a professional finance/stock exchange aesthetic.

## Features

### 1. **Finance Bro Aesthetic**
- **Dark color scheme**: Charcoal backgrounds (#151515), subtle borders (#2A2A2A)
- **Professional colors**: 
  - BUY indicators: Professional green (#00C853)
  - SELL indicators: Professional red (#D32F2F)
  - Accent/headers: Professional blue (#1976D2)
- **Clean layout**: Header bar, bordered panels, alternating row backgrounds
- **Confidence indicators**: Color-coded bars (green/orange/red based on confidence level)

### 2. **Top 10 Recommendations Display**
The main view shows a table with columns:
- **RANK**: Numbered 1-10
- **ITEM**: Bazaar item name (shortened if too long)
- **ACTION**: BUY or SELL (color-coded)
- **CURRENT**: Current price (formatted with K/M suffixes)
- **TARGET**: Predicted price
- **PROFIT**: Expected profit percentage
- **CONF**: Confidence bar with percentage

**Features:**
- Scrollable list (if more than 8 items visible at once)
- Alternating row backgrounds for readability
- Auto-refresh functionality via "Refresh" button
- Powered by your LGBM prediction model

### 3. **Search Bar**
- Located at the top of the GUI
- Type any item name and press ENTER to search
- Shows detailed prediction for that specific item:
  - Item name
  - BUY/SELL recommendation
  - Current and predicted prices
  - Expected profit percentage
  - Confidence level with visual bar
- Press ESC to return to recommendations view

### 4. **Commands**
The GUI can be opened using:
- `/bzgui` - Primary command
- `/bztracker` - Alias

**Note**: `/bzgui` is NOT used in Hypixel Skyblock, so it won't conflict with any existing commands.

## Technical Details

### Files Created/Modified
1. **BazaarTrackerGUI.java** (NEW)
   - Custom GuiScreen implementation
   - Handles rendering, input, and API calls
   - Fetches recommendations from Flask API
   - Implements search functionality

2. **ExampleMod.java** (MODIFIED)
   - Added BazaarGUICommand class
   - Registered new command in init()
   - Added startup message about /bzgui

### API Integration
The GUI connects to your Flask API:
- **GET /recommendations?limit=10&min_confidence=50** - Fetches top recommendations
- **GET /predict/{item_id}** - Fetches prediction for specific item

### Color Scheme Reference
```
Background: #151515 (Dark charcoal)
Panel: #1F1F1F (Lighter charcoal)
Header: #252525 (Header bar)
Border: #2A2A2A (Subtle borders)
Text Primary: #E0E0E0 (Light gray)
Text Secondary: #A0A0A0 (Dimmer gray)
BUY/Green: #00C853 (Professional green)
SELL/Red: #D32F2F (Professional red)
Accent/Blue: #1976D2 (Professional blue)
```

### User Experience
1. User types `/bzgui` in-game
2. GUI opens with dark, professional interface
3. Automatically loads top 10 recommendations
4. User can:
   - Scroll through recommendations (if more than 8)
   - Click "Refresh" to update data
   - Type item name in search bar and press ENTER
   - View detailed prediction for searched item
   - Press ESC to close or return to recommendations

## Usage Instructions

### For Players
1. Make sure your Flask API server is running on port 5001
2. In Minecraft, type `/bzgui` or `/bztracker`
3. Browse the top recommendations or search for specific items
4. Use the scroll wheel to see more recommendations
5. Press ESC to close the GUI

### For Development
- The GUI uses threading to prevent game freezing during API calls
- All network operations are asynchronous
- Error messages are displayed if the API is unreachable
- The GUI doesn't pause the game (`doesGuiPauseGame()` returns false)

## Future Enhancements (Optional)
- Add sorting options (by profit, confidence, etc.)
- Implement item favorites/watchlist
- Add price history charts
- Real-time price updates
- Export recommendations to file
- Configurable color themes
