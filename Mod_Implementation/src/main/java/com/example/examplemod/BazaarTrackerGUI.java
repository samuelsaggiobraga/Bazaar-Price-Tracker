package com.example.examplemod;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.gui.GuiTextField;
import org.lwjgl.input.Keyboard;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;
import java.util.Iterator;

public class BazaarTrackerGUI extends GuiScreen {
    
    private static final String API_URL = "http://localhost:5001";
    private static final int MAX_FAVORITES = 30;
    
    // Colors
    private static final int BACKGROUND_COLOR = 0xE0151515;
    private static final int PANEL_COLOR = 0xE01F1F1F;
    private static final int HEADER_COLOR = 0xE0252525;
    private static final int BORDER_COLOR = 0xFF2A2A2A;
    private static final int TEXT_PRIMARY = 0xFFE0E0E0;
    private static final int TEXT_SECONDARY = 0xFFA0A0A0;
    private static final int BUY_COLOR = 0xFF00C853;
    private static final int SELL_COLOR = 0xFFD32F2F;
    private static final int ACCENT_COLOR = 0xFF1976D2;
    private static final int FAVORITE_COLOR = 0xFFFFD700;
    private static final int AUTOCOMPLETE_BG = 0xE0252525;
    private static final int AUTOCOMPLETE_HOVER = 0xFF3A3A3A;
    
    // View modes
    private enum ViewMode {
        FLIPS, INVESTMENTS, CRASH_WATCH, FAVORITES, SEARCH_RESULT
    }
    
    private enum InvestmentTimeframe {
        ONE_DAY, ONE_WEEK, ONE_MONTH
    }
    
    private ViewMode currentView;
    private InvestmentTimeframe currentTimeframe;
    private GuiTextField searchField;
    private List<RecommendationEntry> flips;
    private List<RecommendationEntry> investments;
    private List<RecommendationEntry> crashWatch;
    private List<RecommendationEntry> favoritePredictions;
    private RecommendationEntry searchResult;
    private Set<String> favoriteItems;
    private List<String> allItemIds;
    private List<String> autocompleteMatches;
    private int selectedAutocomplete;
    private boolean showAutocomplete;

    // New entry-based view state
    private List<EntryRecommendation> currentEntries;  // per-item view
    private String currentItemId;

    // Homescreen ranking: best upcoming positive entry per item
    private List<HomeEntrySummary> homeRankings;
    
    private boolean isLoading;
    private String errorMessage;
    private int scrollOffset;
    private int currentPage;
    private static final int ROWS_PER_PAGE = 8;
    
    // Background polling
    private static Thread pollingThread = null;
    private static volatile boolean shouldPoll = false;
    private static BazaarTrackerGUI activeInstance = null;
    
    // Favorites file
    private File favoritesFile;
    
    public BazaarTrackerGUI() {
        this.flips = new ArrayList<RecommendationEntry>();
        this.investments = new ArrayList<RecommendationEntry>();
        this.crashWatch = new ArrayList<RecommendationEntry>();
        this.favoritePredictions = new ArrayList<RecommendationEntry>();
        this.favoriteItems = new HashSet<String>();
        this.allItemIds = new ArrayList<String>();
        this.autocompleteMatches = new ArrayList<String>();
        this.currentView = ViewMode.FLIPS;
        this.currentTimeframe = InvestmentTimeframe.ONE_DAY;
        this.currentEntries = new ArrayList<EntryRecommendation>();
        this.currentItemId = null;
        this.homeRankings = new ArrayList<HomeEntrySummary>();
        this.isLoading = false;
        this.scrollOffset = 0;
        this.currentPage = 0;
        this.selectedAutocomplete = -1;
        this.showAutocomplete = false;
        
        try {
            this.favoritesFile = new File(Minecraft.getMinecraft().mcDataDir, "bazaar_favorites.txt");
            loadFavorites();
            fetchAllItems();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    @Override
    public void initGui() {
        super.initGui();
        Keyboard.enableRepeatEvents(true);
        
        int centerX = this.width / 2;
        int searchBarWidth = 200;
        
        // Clear existing buttons
        this.buttonList.clear();
        
        // Search field (logic kept but not shown)
        this.searchField = new GuiTextField(0, this.fontRendererObj, centerX - 100, 15, searchBarWidth, 16);
        this.searchField.setMaxStringLength(50);
        this.searchField.setFocused(false);
        this.searchField.setText("");
        
        // Single refresh button
        this.buttonList.add(new GuiButton(1, 10, 10, 70, 20, "Refresh"));

        // Page navigation buttons in bottom-right
        int navY = this.height - 40;
        this.buttonList.add(new GuiButton(2, this.width - 60, navY, 20, 20, "<"));
        this.buttonList.add(new GuiButton(3, this.width - 35, navY, 20, 20, ">"));

        // On first open, load homescreen rankings
        if (homeRankings == null || homeRankings.isEmpty()) {
            fetchHomeRankings();
        }
    }
    
    @Override
    public void updateScreen() {
        super.updateScreen();
    }
    
    @Override
    public void onGuiClosed() {
        Keyboard.enableRepeatEvents(false);
        stopPolling();
        activeInstance = null;
    }
    
    @Override
    protected void actionPerformed(GuiButton button) {
        if (button.id == 1) {
            // Refresh homescreen rankings
            scrollOffset = 0;
            currentPage = 0;
            fetchHomeRankings();
        } else if (button.id == 2) {
            // Previous page
            if (currentPage > 0) {
                currentPage--;
            }
        } else if (button.id == 3) {
            // Next page
            int total = homeRankings != null ? homeRankings.size() : 0;
            if (total > 0) {
                int maxPage = (total - 1) / ROWS_PER_PAGE;
                if (currentPage < maxPage) {
                    currentPage++;
                }
            }
        }
    }
    
    @Override
    protected void keyTyped(char typedChar, int keyCode) {
        if (this.searchField.isFocused()) {
            String before = this.searchField.getText();
            this.searchField.textboxKeyTyped(typedChar, keyCode);
            String after = this.searchField.getText();
            
            // Update autocomplete on text change
            if (!before.equals(after)) {
                updateAutocomplete(after);
            }
            
            // Handle autocomplete navigation
            if (showAutocomplete && !autocompleteMatches.isEmpty()) {
                if (keyCode == Keyboard.KEY_DOWN) {
                    selectedAutocomplete = Math.min(selectedAutocomplete + 1, autocompleteMatches.size() - 1);
                    return;
                } else if (keyCode == Keyboard.KEY_UP) {
                    selectedAutocomplete = Math.max(selectedAutocomplete - 1, 0);
                    return;
                }
            }
            
            // Search on Enter
            if (keyCode == Keyboard.KEY_RETURN) {
                if (showAutocomplete && selectedAutocomplete >= 0 && selectedAutocomplete < autocompleteMatches.size()) {
                    // Use selected autocomplete
                    String selected = autocompleteMatches.get(selectedAutocomplete);
                    this.searchField.setText(selected);
                    showAutocomplete = false;
                    searchItem(selected);
                } else {
                    String query = this.searchField.getText().trim();
                    if (!query.isEmpty()) {
                        searchItem(query);
                    }
                }
            } else if (keyCode == Keyboard.KEY_ESCAPE) {
                if (showAutocomplete) {
                    showAutocomplete = false;
                } else {
                    this.mc.displayGuiScreen(null);
                }
            }
        } else if (keyCode == Keyboard.KEY_ESCAPE) {
            this.mc.displayGuiScreen(null);
        }
    }
    
    @Override
    protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws java.io.IOException {
        super.mouseClicked(mouseX, mouseY, mouseButton);
    }
    
    @Override
    public void handleMouseInput() throws java.io.IOException {
        // Scroll disabled in favor of page-based navigation
        super.handleMouseInput();
    }
    
    @Override
    public void drawScreen(int mouseX, int mouseY, float partialTicks) {
        // Draw dark background
        drawRect(0, 0, this.width, this.height, BACKGROUND_COLOR);
        
        int centerX = this.width / 2;
        int panelWidth = Math.min(600, this.width - 40);
        int panelHeight = Math.min(400, this.height - 100);
        int panelX = centerX - panelWidth / 2;
        int panelY = 50;
        
        // Main panel
        drawRect(panelX, panelY, panelX + panelWidth, panelY + panelHeight, PANEL_COLOR);
        drawRect(panelX, panelY, panelX + panelWidth, panelY + 1, BORDER_COLOR);
        drawRect(panelX, panelY + panelHeight - 1, panelX + panelWidth, panelY + panelHeight, BORDER_COLOR);
        drawRect(panelX, panelY, panelX + 1, panelY + panelHeight, BORDER_COLOR);
        drawRect(panelX + panelWidth - 1, panelY, panelX + panelWidth, panelY + panelHeight, BORDER_COLOR);
        
        // Header
        drawRect(panelX, panelY, panelX + panelWidth, panelY + 35, HEADER_COLOR);
        drawRect(panelX, panelY + 35, panelX + panelWidth, panelY + 36, BORDER_COLOR);
        
        String title;
        if (currentItemId != null && !currentItemId.isEmpty()) {
            title = "ENTRY SIGNALS - " + currentItemId.replace("_", " ");
        } else {
            title = "BAZAAR ENTRY SIGNALS";
        }
        
        drawCenteredString(this.fontRendererObj, title, centerX, panelY + 13, TEXT_PRIMARY);
        
        // Search UI removed for entry signals homescreen
        
        // Draw content
        if (isLoading) {
            drawCenteredString(this.fontRendererObj, "Loading...", centerX, panelY + 220, TEXT_SECONDARY);
        } else if (errorMessage != null) {
            drawCenteredString(this.fontRendererObj, "Error: " + errorMessage, centerX, panelY + 220, SELL_COLOR);
        } else if (currentEntries != null && !currentEntries.isEmpty()) {
            // Detailed per-item view after a search
            drawEntries(panelX, panelY, panelWidth, panelHeight);
        } else if (homeRankings != null && !homeRankings.isEmpty()) {
            // Homescreen: ranking of items by closest positive prediction
            drawHomeRankings(panelX, panelY, panelWidth, panelHeight);
        } else {
            drawCenteredString(this.fontRendererObj, "Type an item name and press Enter to see entry signals", centerX, panelY + 220, TEXT_SECONDARY);
        }
        
        super.drawScreen(mouseX, mouseY, partialTicks);
        
        drawCenteredString(this.fontRendererObj, "Powered by LGBM Entry Model", centerX, this.height - 15, TEXT_SECONDARY);
    }
    
    private void drawAutocomplete(int x, int y) {
        int width = 300;
        int maxVisible = Math.min(5, autocompleteMatches.size());
        int height = maxVisible * 20;
        
        // Background
        drawRect(x, y, x + width, y + height, AUTOCOMPLETE_BG);
        drawRect(x, y, x + width, y + 1, BORDER_COLOR);
        drawRect(x, y + height - 1, x + width, y + height, BORDER_COLOR);
        drawRect(x, y, x + 1, y + height, BORDER_COLOR);
        drawRect(x + width - 1, y, x + width, y + height, BORDER_COLOR);
        
        for (int i = 0; i < maxVisible; i++) {
            String item = autocompleteMatches.get(i).replace("_", " ");
            int itemY = y + i * 20;
            
            // Highlight selected
            if (i == selectedAutocomplete) {
                drawRect(x + 1, itemY, x + width - 1, itemY + 20, AUTOCOMPLETE_HOVER);
            }
            
            drawString(this.fontRendererObj, item, x + 5, itemY + 6, TEXT_PRIMARY);
        }
    }

    /**
     * Draw the entry-based recommendations (best timestamps to enter a trade).
     */
    private void drawEntries(int panelX, int panelY, int panelWidth, int panelHeight) {
        int headerY = panelY + 50;
        int rowHeight = 40;

        int colRankX = panelX + 25;
        int colTimeX = panelX + 80;
        int colBuyX = panelX + 220;
        int colSellX = panelX + 330;
        int colScoreX = panelX + 455;

        drawString(this.fontRendererObj, "#", colRankX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ENTRY TIME", colTimeX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "BUY PRICE", colBuyX, headerY, BUY_COLOR);
        drawString(this.fontRendererObj, "SELL PRICE", colSellX, headerY, SELL_COLOR);
        drawString(this.fontRendererObj, "ENTRY SCORE", colScoreX, headerY, ACCENT_COLOR);

        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);

        int startY = headerY + 20;
        int total = currentEntries != null ? currentEntries.size() : 0;
        int visibleItems = Math.min(8, total - scrollOffset);

        for (int i = 0; i < visibleItems; i++) {
            int index = i + scrollOffset;
            if (index >= total) break;

            EntryRecommendation rec = currentEntries.get(index);
            int rowY = startY + i * rowHeight;

            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }

            drawString(this.fontRendererObj, "#" + (index + 1), colRankX, rowY + 5, TEXT_PRIMARY);

            String timeStr = rec.timestamp != null ? rec.timestamp : "";
            if (timeStr.length() > 19) timeStr = timeStr.substring(0, 19);
            drawString(this.fontRendererObj, timeStr, colTimeX, rowY + 5, TEXT_PRIMARY);

            drawString(this.fontRendererObj, formatPrice(rec.buyPrice), colBuyX, rowY + 5, TEXT_PRIMARY);
            drawString(this.fontRendererObj, formatPrice(rec.sellPrice), colSellX, rowY + 5, TEXT_PRIMARY);

            String scoreStr = String.format("%.4f", rec.entryScore);
            drawString(this.fontRendererObj, scoreStr, colScoreX, rowY + 5, ACCENT_COLOR);
        }

        if (total > 8) {
            String scrollText = String.format("Showing %d-%d of %d", scrollOffset + 1,
                Math.min(scrollOffset + 8, total), total);
            drawCenteredString(this.fontRendererObj, scrollText, panelX + panelWidth / 2,
                panelY + panelHeight - 20, TEXT_SECONDARY);
        }
    }
    
    private void drawFavorites(int panelX, int panelY, int panelWidth, int panelHeight) {
        int centerX = this.width / 2;
        
        if (favoriteItems.isEmpty()) {
            drawCenteredString(this.fontRendererObj, "No favorites yet!", centerX, panelY + 200, TEXT_SECONDARY);
            drawCenteredString(this.fontRendererObj, "Click the * next to items to add favorites (max 30)", 
                centerX, panelY + 220, TEXT_SECONDARY);
            return;
        }
        
        drawCenteredString(this.fontRendererObj, String.format("Tracking %d/%d favorites", 
            favoriteItems.size(), MAX_FAVORITES), centerX, panelY + 45, ACCENT_COLOR);
        
        int headerY = panelY + 65;
        int rowHeight = 55; // Match recommendations layout
        
        int colStarX = panelX + 15;
        int colItemX = panelX + 50;
        int colBuyX = panelX + 210;
        int colSellX = panelX + 335;
        int colSpreadX = panelX + 460;
        
        drawString(this.fontRendererObj, "*", colStarX, headerY, FAVORITE_COLOR);
        drawString(this.fontRendererObj, "ITEM", colItemX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "BUY PRICE", colBuyX, headerY, BUY_COLOR);
        drawString(this.fontRendererObj, "SELL PRICE", colSellX, headerY, SELL_COLOR);
        drawString(this.fontRendererObj, "SPREAD", colSpreadX, headerY, ACCENT_COLOR);
        
        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);
        
        int startY = headerY + 20;
        int visibleItems = Math.min(5, favoritePredictions.size() - scrollOffset); // Reduced for larger rows
        
        for (int i = 0; i < visibleItems; i++) {
            int index = i + scrollOffset;
            if (index >= favoritePredictions.size()) break;
            
            RecommendationEntry rec = favoritePredictions.get(index);
            int rowY = startY + i * rowHeight;
            
            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }
            
            drawString(this.fontRendererObj, "*", colStarX, rowY + 2, FAVORITE_COLOR);
            
            String itemName = rec.itemId.replace("_", " ");
            if (itemName.length() > 20) itemName = itemName.substring(0, 17) + "...";
            drawString(this.fontRendererObj, itemName, colItemX, rowY + 5, TEXT_PRIMARY);
            
            // BUY PRICE - Line 1: Current, Line 2: Predicted with arrow
            if (rec.buyCurrentPrice > 0) {
                drawString(this.fontRendererObj, formatPrice(rec.buyCurrentPrice), colBuyX, rowY + 2, TEXT_PRIMARY);
                String arrow = rec.buyDirection.equals("UP") ? "\u2191" : "\u2193";
                int arrowColor = rec.buyDirection.equals("UP") ? BUY_COLOR : SELL_COLOR;
                drawString(this.fontRendererObj, arrow + formatPrice(rec.buyPredictedPrice), colBuyX, rowY + 12, arrowColor);
            } else {
                drawString(this.fontRendererObj, formatPrice(rec.currentPrice), colBuyX, rowY + 2, TEXT_PRIMARY);
                drawString(this.fontRendererObj, "--", colBuyX, rowY + 12, TEXT_SECONDARY);
            }
            
            // SELL PRICE - Line 1: Current, Line 2: Predicted with arrow
            if (rec.sellCurrentPrice > 0) {
                drawString(this.fontRendererObj, formatPrice(rec.sellCurrentPrice), colSellX, rowY + 2, TEXT_PRIMARY);
                String arrow = rec.sellDirection.equals("UP") ? "\u2191" : "\u2193";
                int arrowColor = rec.sellDirection.equals("UP") ? BUY_COLOR : SELL_COLOR;
                drawString(this.fontRendererObj, arrow + formatPrice(rec.sellPredictedPrice), colSellX, rowY + 12, arrowColor);
            } else {
                drawString(this.fontRendererObj, formatPrice(rec.currentPrice), colSellX, rowY + 2, TEXT_PRIMARY);
                drawString(this.fontRendererObj, "--", colSellX, rowY + 12, TEXT_SECONDARY);
            }
            
            // SPREAD - Line 1: Current (absolute), Line 2: Predicted change in absolute spread
            if (rec.spreadCurrent != 0 && rec.spreadPredicted != 0) {
                // Display ABSOLUTE spread value (always positive for readability)
                drawString(this.fontRendererObj, formatPrice(Math.abs(rec.spreadCurrent)), colSpreadX, rowY + 2, TEXT_PRIMARY);
                
                // Calculate change in ABSOLUTE spread
                double absSpreadCurrent = Math.abs(rec.spreadCurrent);
                double absSpreadPredicted = Math.abs(rec.spreadPredicted);
                double absSpreadChangePct = ((absSpreadPredicted - absSpreadCurrent) / absSpreadCurrent) * 100;
                
                // WIDEN = larger absolute spread = MORE profit = GOOD (green up)
                // NARROW = smaller absolute spread = LESS profit = BAD (red down)
                String arrow = rec.spreadDirection.equals("WIDEN") ? "\u2191" : "\u2193";
                int arrowColor = rec.spreadDirection.equals("WIDEN") ? BUY_COLOR : SELL_COLOR;
                
                String spreadText = String.format("%s%.1f%%", arrow, Math.abs(absSpreadChangePct));
                drawString(this.fontRendererObj, spreadText, colSpreadX, rowY + 12, arrowColor);
            } else {
                double spread = rec.currentPrice * 0.02; // Fallback estimate
                drawString(this.fontRendererObj, formatPrice(spread), colSpreadX, rowY + 2, TEXT_PRIMARY);
                drawString(this.fontRendererObj, "--", colSpreadX, rowY + 12, TEXT_SECONDARY);
            }
        }
    }
    
    private void drawSearchResult(int panelX, int panelY) {
        int centerX = this.width / 2;
        int startY = panelY + 60;
        int leftCol = centerX - 180;
        int rightCol = centerX + 20;
        
        drawCenteredString(this.fontRendererObj, "COMPREHENSIVE PREDICTION", centerX, startY, ACCENT_COLOR);
        
        String itemName = searchResult.itemId.replace("_", " ");
        
        // Favorite star and item name
        boolean isFav = favoriteItems.contains(searchResult.itemId);
        drawString(this.fontRendererObj, isFav ? "*" : "o", leftCol - 15, startY + 21, 
            isFav ? FAVORITE_COLOR : TEXT_SECONDARY);
        drawCenteredString(this.fontRendererObj, itemName, centerX, startY + 20, TEXT_PRIMARY);
        
        // Overall recommendation
        if (searchResult.recommendation != null && !searchResult.recommendation.isEmpty()) {
            String rec = searchResult.recommendation;
            int recColor = TEXT_SECONDARY;  // Default
            
            if (rec.contains("STRONG_BUY") || rec.equals("ARBITRAGE")) {
                recColor = 0xFF00E676;  // Bright green
            } else if (rec.contains("BUY") && !rec.contains("SELL")) {
                recColor = BUY_COLOR;
            } else if (rec.contains("STRONG_SELL")) {
                recColor = 0xFFE53935;  // Bright red
            } else if (rec.contains("SELL") && !rec.contains("BUY")) {
                recColor = SELL_COLOR;
            } else if (rec.equals("WAIT")) {
                recColor = 0xFFFFA726;  // Orange
            }
            
            drawCenteredString(this.fontRendererObj, "=> " + rec + " <=", centerX, startY + 40, recColor);
        }
        
        int y = startY + 65;
        
        // BUY PRICE (Sell Order Price)
        if (searchResult.buyCurrentPrice > 0) {
            drawString(this.fontRendererObj, "SELL ORDER PRICE:", leftCol, y, BUY_COLOR);
            drawString(this.fontRendererObj, formatPrice(searchResult.buyCurrentPrice), leftCol + 110, y, TEXT_PRIMARY);
            
            String arrow = searchResult.buyDirection.equals("UP") ? "\u2191" : "\u2193";
            int arrowColor = searchResult.buyDirection.equals("UP") ? BUY_COLOR : SELL_COLOR;
            drawString(this.fontRendererObj, arrow + formatPrice(searchResult.buyPredictedPrice), leftCol + 110, y + 10, arrowColor);
            
            drawString(this.fontRendererObj, String.format("%.1f%% (%.0f%% conf)", 
                searchResult.buyChangePct, searchResult.buyConfidence), leftCol + 110, y + 20, TEXT_SECONDARY);
            y += 35;
        }
        
        // SELL PRICE (Buy Order Price)
        if (searchResult.sellCurrentPrice > 0) {
            drawString(this.fontRendererObj, "BUY ORDER PRICE:", leftCol, y, SELL_COLOR);
            drawString(this.fontRendererObj, formatPrice(searchResult.sellCurrentPrice), leftCol + 110, y, TEXT_PRIMARY);
            
            String arrow = searchResult.sellDirection.equals("UP") ? "\u2191" : "\u2193";
            int arrowColor = searchResult.sellDirection.equals("UP") ? BUY_COLOR : SELL_COLOR;
            drawString(this.fontRendererObj, arrow + formatPrice(searchResult.sellPredictedPrice), leftCol + 110, y + 10, arrowColor);
            
            drawString(this.fontRendererObj, String.format("%.1f%% (%.0f%% conf)", 
                searchResult.sellChangePct, searchResult.sellConfidence), leftCol + 110, y + 20, TEXT_SECONDARY);
            y += 35;
        }
        
        // SPREAD ANALYSIS (ML-predicted spread change)
        if (searchResult.spreadCurrent != 0 && searchResult.spreadPredicted != 0) {
            drawString(this.fontRendererObj, "SPREAD:", leftCol, y, ACCENT_COLOR);
            // Display ABSOLUTE spread (always positive)
            drawString(this.fontRendererObj, formatPrice(Math.abs(searchResult.spreadCurrent)), leftCol + 110, y, TEXT_PRIMARY);
            
            // Calculate change in ABSOLUTE spread from ML prediction
            double absSpreadCurrent = Math.abs(searchResult.spreadCurrent);
            double absSpreadPredicted = Math.abs(searchResult.spreadPredicted);
            double absSpreadChangePct = ((absSpreadPredicted - absSpreadCurrent) / absSpreadCurrent) * 100;
            
            // WIDEN = larger absolute spread = MORE profit = GOOD (green up)
            // NARROW = smaller absolute spread = LESS profit = BAD (red down)
            String arrow = searchResult.spreadDirection.equals("WIDEN") ? "\u2191" : "\u2193";
            int arrowColor = searchResult.spreadDirection.equals("WIDEN") ? BUY_COLOR : SELL_COLOR;
            
            // Show absolute spread change % from ML model
            drawString(this.fontRendererObj, String.format("%s%.1f%% (%s)", arrow, Math.abs(absSpreadChangePct), 
                searchResult.spreadDirection), leftCol + 110, y + 10, arrowColor);
            
            drawString(this.fontRendererObj, String.format("Confidence: %.0f%%", searchResult.spreadConfidence), 
                leftCol + 110, y + 20, TEXT_SECONDARY);
            y += 35;
        }
        
        // FLIP PROFIT POTENTIAL (Sell Order - Buy Order) / Buy Order
        if (searchResult.buyCurrentPrice > 0 && searchResult.sellCurrentPrice > 0) {
            drawString(this.fontRendererObj, "FLIP PROFIT:", leftCol, y, ACCENT_COLOR);
            
            // Calculate: (sell_order - buy_order) / buy_order * 100
            // buy_current = sell order price, sell_current = buy order price (API reversal)
            double currentFlipProfit = ((searchResult.buyCurrentPrice - searchResult.sellCurrentPrice) / searchResult.sellCurrentPrice) * 100;
            double predictedFlipProfit = ((searchResult.buyPredictedPrice - searchResult.sellPredictedPrice) / searchResult.sellPredictedPrice) * 100;
            
            String currentFlip = String.format("Now: %.2f%%", currentFlipProfit);
            int currentColor = currentFlipProfit > 0 ? BUY_COLOR : SELL_COLOR;
            drawString(this.fontRendererObj, currentFlip, leftCol + 110, y, currentColor);
            
            String predictedFlip = String.format("Pred: %.2f%%", predictedFlipProfit);
            int predictedColor = predictedFlipProfit > 0 ? BUY_COLOR : SELL_COLOR;
            drawString(this.fontRendererObj, predictedFlip, leftCol + 110, y + 10, predictedColor);
            
            y += 25;
        }
        
        // Footer note
        drawCenteredString(this.fontRendererObj, "Press ESC to return | Click * to favorite", 
            centerX, panelY + 380, TEXT_SECONDARY);
    }
    
    private void drawFlips(int panelX, int panelY, int panelWidth, int panelHeight) {
        int headerY = panelY + 50;
        int rowHeight = 40;
        
        int colStarX = panelX + 10;
        int colRankX = panelX + 25;
        int colItemX = panelX + 60;
        int colBuyX = panelX + 230;
        int colSellX = panelX + 340;
        int colSpreadX = panelX + 450;
        
        drawString(this.fontRendererObj, "*", colStarX, headerY, FAVORITE_COLOR);
        drawString(this.fontRendererObj, "#", colRankX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ITEM", colItemX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "BUY ORDER", colBuyX, headerY, SELL_COLOR);
        drawString(this.fontRendererObj, "SELL ORDER", colSellX, headerY, BUY_COLOR);
        drawString(this.fontRendererObj, "SPREAD %", colSpreadX, headerY, ACCENT_COLOR);
        
        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);
        
        int startY = headerY + 20;
        int visibleItems = Math.min(8, flips.size() - scrollOffset);
        
        for (int i = 0; i < visibleItems; i++) {
            int index = i + scrollOffset;
            if (index >= flips.size()) break;
            
            RecommendationEntry rec = flips.get(index);
            int rowY = startY + i * rowHeight;
            
            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }
            
            // Star for favorites
            boolean isFav = favoriteItems.contains(rec.itemId);
            drawString(this.fontRendererObj, isFav ? "*" : "o", colStarX, rowY + 5, 
                isFav ? FAVORITE_COLOR : TEXT_SECONDARY);
            
            drawString(this.fontRendererObj, "#" + (index + 1), colRankX, rowY + 5, TEXT_PRIMARY);
            
            String itemName = rec.itemId.replace("_", " ");
            if (itemName.length() > 20) itemName = itemName.substring(0, 17) + "...";
            drawString(this.fontRendererObj, itemName, colItemX, rowY + 5, TEXT_PRIMARY);
            
            drawString(this.fontRendererObj, formatPrice(rec.buyOrderPrice), colBuyX, rowY + 5, TEXT_PRIMARY);
            drawString(this.fontRendererObj, formatPrice(rec.sellOrderPrice), colSellX, rowY + 5, TEXT_PRIMARY);
            
            String spreadStr = String.format("%.1f%%", rec.spreadPct);
            int spreadColor = rec.spreadPct > 10 ? 0xFF00E676 : (rec.spreadPct > 5 ? BUY_COLOR : ACCENT_COLOR);
            drawString(this.fontRendererObj, spreadStr, colSpreadX, rowY + 5, spreadColor);
        }
        
        if (flips.size() > 8) {
            String scrollText = String.format("Showing %d-%d of %d", scrollOffset + 1, 
                Math.min(scrollOffset + 8, flips.size()), flips.size());
            drawCenteredString(this.fontRendererObj, scrollText, panelX + panelWidth / 2, 
                panelY + panelHeight - 20, TEXT_SECONDARY);
        }
    }
    
    private void drawInvestments(int panelX, int panelY, int panelWidth, int panelHeight) {
        int headerY = panelY + 50;
        int rowHeight = 40;
        
        int colStarX = panelX + 10;
        int colRankX = panelX + 25;
        int colItemX = panelX + 60;
        int colCurrentX = panelX + 230;
        int colTargetX = panelX + 320;
        int colReturnX = panelX + 410;
        int colConfX = panelX + 500;
        
        drawString(this.fontRendererObj, "*", colStarX, headerY, FAVORITE_COLOR);
        drawString(this.fontRendererObj, "#", colRankX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ITEM", colItemX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "CURRENT", colCurrentX, headerY, TEXT_PRIMARY);
        drawString(this.fontRendererObj, "TARGET", colTargetX, headerY, BUY_COLOR);
        drawString(this.fontRendererObj, "W. RETURN", colReturnX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "CONF", colConfX, headerY, ACCENT_COLOR);
        
        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);
        
        int startY = headerY + 20;
        int visibleItems = Math.min(8, investments.size() - scrollOffset);
        
        for (int i = 0; i < visibleItems; i++) {
            int index = i + scrollOffset;
            if (index >= investments.size()) break;
            
            RecommendationEntry rec = investments.get(index);
            int rowY = startY + i * rowHeight;
            
            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }
            
            // Star for favorites
            boolean isFav = favoriteItems.contains(rec.itemId);
            drawString(this.fontRendererObj, isFav ? "*" : "o", colStarX, rowY + 5, 
                isFav ? FAVORITE_COLOR : TEXT_SECONDARY);
            
            drawString(this.fontRendererObj, "#" + (index + 1), colRankX, rowY + 5, TEXT_PRIMARY);
            
            String itemName = rec.itemId.replace("_", " ");
            if (itemName.length() > 20) itemName = itemName.substring(0, 17) + "...";
            drawString(this.fontRendererObj, itemName, colItemX, rowY + 5, TEXT_PRIMARY);
            
            drawString(this.fontRendererObj, formatPrice(rec.currentPrice), colCurrentX, rowY + 5, TEXT_PRIMARY);
            drawString(this.fontRendererObj, formatPrice(rec.predictedPrice), colTargetX, rowY + 5, BUY_COLOR);
            
            String returnStr = String.format("%.2f", rec.weightedReturn);
            drawString(this.fontRendererObj, returnStr, colReturnX, rowY + 5, BUY_COLOR);
            
            String confStr = String.format("%.0f%%", rec.confidence);
            drawString(this.fontRendererObj, confStr, colConfX, rowY + 5, TEXT_SECONDARY);
        }
        
        if (investments.size() > 8) {
            String scrollText = String.format("Showing %d-%d of %d", scrollOffset + 1, 
                Math.min(scrollOffset + 8, investments.size()), investments.size());
            drawCenteredString(this.fontRendererObj, scrollText, panelX + panelWidth / 2, 
                panelY + panelHeight - 20, TEXT_SECONDARY);
        }
    }
    
    private void drawCrashWatch(int panelX, int panelY, int panelWidth, int panelHeight) {
        int headerY = panelY + 50;
        int rowHeight = 40;
        
        int colStarX = panelX + 10;
        int colRankX = panelX + 25;
        int colItemX = panelX + 60;
        int colCurrentX = panelX + 210;
        int colCrashX = panelX + 290;
        int colConfX = panelX + 360;
        int colReversalX = panelX + 420;
        int colActionX = panelX + 490;
        
        drawString(this.fontRendererObj, "*", colStarX, headerY, FAVORITE_COLOR);
        drawString(this.fontRendererObj, "#", colRankX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ITEM", colItemX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "PRICE", colCurrentX, headerY, TEXT_PRIMARY);
        drawString(this.fontRendererObj, "CRASH %", colCrashX, headerY, SELL_COLOR);
        drawString(this.fontRendererObj, "CONF", colConfX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "REV", colReversalX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ACTION", colActionX, headerY, ACCENT_COLOR);
        
        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);
        
        int startY = headerY + 20;
        int visibleItems = Math.min(8, crashWatch.size() - scrollOffset);
        
        for (int i = 0; i < visibleItems; i++) {
            int index = i + scrollOffset;
            if (index >= crashWatch.size()) break;
            
            RecommendationEntry rec = crashWatch.get(index);
            int rowY = startY + i * rowHeight;
            
            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }
            
            // Star for favorites
            boolean isFav = favoriteItems.contains(rec.itemId);
            drawString(this.fontRendererObj, isFav ? "*" : "o", colStarX, rowY + 5, 
                isFav ? FAVORITE_COLOR : TEXT_SECONDARY);
            
            drawString(this.fontRendererObj, "#" + (index + 1), colRankX, rowY + 5, TEXT_PRIMARY);
            
            String itemName = rec.itemId.replace("_", " ");
            if (itemName.length() > 18) itemName = itemName.substring(0, 15) + "...";
            drawString(this.fontRendererObj, itemName, colItemX, rowY + 5, TEXT_PRIMARY);
            
            drawString(this.fontRendererObj, formatPrice(rec.currentPrice), colCurrentX, rowY + 5, TEXT_PRIMARY);
            
            String crashStr = String.format("%.1f%%", Math.abs(rec.crashPct));
            drawString(this.fontRendererObj, crashStr, colCrashX, rowY + 5, SELL_COLOR);
            
            String confStr = String.format("%.0f%%", rec.confidence);
            drawString(this.fontRendererObj, confStr, colConfX, rowY + 5, TEXT_SECONDARY);
            
            String reversalStr = rec.reversalHours + "h";
            int reversalColor = rec.reversalHours <= 12 ? BUY_COLOR : 0xFFFFA726;
            drawString(this.fontRendererObj, reversalStr, colReversalX, rowY + 5, reversalColor);
            
            String action = rec.recommendation != null ? rec.recommendation : "WAIT";
            int actionColor = action.equals("BUY_DIP") ? BUY_COLOR : 0xFFFFA726;
            drawString(this.fontRendererObj, action, colActionX, rowY + 5, actionColor);
        }
        
        if (crashWatch.size() > 8) {
            String scrollText = String.format("Showing %d-%d of %d", scrollOffset + 1, 
                Math.min(scrollOffset + 8, crashWatch.size()), crashWatch.size());
            drawCenteredString(this.fontRendererObj, scrollText, panelX + panelWidth / 2, 
                panelY + panelHeight - 20, TEXT_SECONDARY);
        }
    }
    
    private int getConfidenceColor(double confidence) {
        if (confidence >= 80) return 0xFF00C853;
        if (confidence >= 60) return 0xFFFFA726;
        return 0xFFFF5252;
    }
    
    private String formatPrice(double price) {
        if (price >= 1000000) return String.format("%.2fM", price / 1000000);
        if (price >= 1000) return String.format("%.2fK", price / 1000);
        return String.format("%.2f", price);
    }
    
    private void updateAutocomplete(String query) {
        if (query.trim().isEmpty()) {
            showAutocomplete = false;
            autocompleteMatches.clear();
            return;
        }
        
        String queryLower = query.toLowerCase().replace(" ", "_");
        autocompleteMatches.clear();
        
        for (String itemId : allItemIds) {
            if (itemId.toLowerCase().contains(queryLower)) {
                autocompleteMatches.add(itemId);
                if (autocompleteMatches.size() >= 5) break;
            }
        }
        
        showAutocomplete = !autocompleteMatches.isEmpty();
        selectedAutocomplete = showAutocomplete ? 0 : -1;
    }
    
    private void toggleFavorite(String itemId) {
        if (favoriteItems.contains(itemId)) {
            favoriteItems.remove(itemId);
            // Remove matching entries manually (Java 6 compatible)
            Iterator<RecommendationEntry> iterator = favoritePredictions.iterator();
            while (iterator.hasNext()) {
                RecommendationEntry e = iterator.next();
                if (e.itemId.equals(itemId)) {
                    iterator.remove();
                }
            }
        } else {
            if (favoriteItems.size() < MAX_FAVORITES) {
                favoriteItems.add(itemId);
            }
        }
        saveFavorites();
    }
    
    private void loadFavorites() {
        try {
            if (favoritesFile != null && favoritesFile.exists()) {
                BufferedReader reader = new BufferedReader(new FileReader(favoritesFile));
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty()) {
                        favoriteItems.add(line);
                    }
                }
                reader.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private void saveFavorites() {
        try {
            if (favoritesFile == null) return;
            BufferedWriter writer = new BufferedWriter(new FileWriter(favoritesFile));
            for (String itemId : favoriteItems) {
                writer.write(itemId);
                writer.newLine();
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private void fetchAllItems() {
        new Thread(new Runnable() {
            public void run() {
                try {
                    String response = fetchFromUrl("https://sky.coflnet.com/api/items/bazaar/tags");
                    JsonArray items = new JsonParser().parse(response).getAsJsonArray();
                    for (int i = 0; i < items.size(); i++) {
                        allItemIds.add(items.get(i).getAsString());
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
    
    private void fetchFlips() {
        this.isLoading = true;
        this.errorMessage = null;
        
        new Thread(new Runnable() {
            public void run() {
                try {
                    String response = fetchFromUrl(API_URL + "/flips?limit=100");
                    parseFlips(response);
                    isLoading = false;
                } catch (Exception e) {
                    errorMessage = e.getMessage();
                    isLoading = false;
                }
            }
        }).start();
    }
    
    private void fetchInvestments() {
        this.isLoading = true;
        this.errorMessage = null;
        
        new Thread(new Runnable() {
            public void run() {
                try {
                    String timeframe = "1d";
                    if (currentTimeframe == InvestmentTimeframe.ONE_WEEK) timeframe = "1w";
                    else if (currentTimeframe == InvestmentTimeframe.ONE_MONTH) timeframe = "1m";
                    
                    String response = fetchFromUrl(API_URL + "/investments?timeframe=" + timeframe + "&limit=100");
                    parseInvestments(response);
                    isLoading = false;
                } catch (Exception e) {
                    errorMessage = e.getMessage();
                    isLoading = false;
                }
            }
        }).start();
    }
    
    private void fetchCrashWatch() {
        this.isLoading = true;
        this.errorMessage = null;
        
        new Thread(new Runnable() {
            public void run() {
                try {
                    String response = fetchFromUrl(API_URL + "/crash_watch?limit=100");
                    parseCrashWatch(response);
                    isLoading = false;
                } catch (Exception e) {
                    errorMessage = e.getMessage();
                    isLoading = false;
                }
            }
        }).start();
    }
    
    private void searchItem(final String itemId) {
        this.isLoading = true;
        this.errorMessage = null;
        this.currentView = ViewMode.SEARCH_RESULT;
        this.searchResult = null;
        this.showAutocomplete = false;
        
        new Thread(new Runnable() {
            public void run() {
                try {
                    String cleanId = itemId.toUpperCase().replace(" ", "_");
                    String response = fetchFromUrl(API_URL + "/predict/" + cleanId);
                    parseSearchResult(response);
                    isLoading = false;
                } catch (Exception e) {
                    String msg = e.getMessage();
                    if (msg != null && msg.contains("404")) {
                        errorMessage = "Item not found: " + itemId;
                    } else if (msg != null && msg.contains("503")) {
                        errorMessage = "Server not ready: models are still loading or missing";
                    } else {
                        errorMessage = "Error contacting prediction server: " + msg;
                    }
                    isLoading = false;
                }
            }
        }).start();
    }
    
    private String fetchFromUrl(String urlString) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(5000);
        conn.setReadTimeout(5000);
        
        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            throw new Exception("Server returned code: " + responseCode);
        }
        
        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder response = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            response.append(line);
        }
        in.close();
        return response.toString();
    }
    
    private void parseFlips(String jsonResponse) {
        try {
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(jsonResponse).getAsJsonObject();
            JsonArray flipsArray = root.getAsJsonArray("flips");
            
            List<RecommendationEntry> newFlips = new ArrayList<RecommendationEntry>();
            for (int i = 0; i < flipsArray.size(); i++) {
                JsonObject flip = flipsArray.get(i).getAsJsonObject();
                RecommendationEntry entry = new RecommendationEntry();
                entry.itemId = flip.get("item_id").getAsString();
                entry.buyOrderPrice = flip.get("buy_order_price").getAsDouble();
                entry.sellOrderPrice = flip.get("sell_order_price").getAsDouble();
                entry.spreadValue = flip.get("spread").getAsDouble();
                entry.spreadPct = flip.get("spread_pct").getAsDouble();
                entry.buyDirection = flip.get("buy_direction").getAsString();
                entry.sellDirection = flip.get("sell_direction").getAsString();
                entry.spreadDirection = flip.get("spread_direction").getAsString();
                newFlips.add(entry);
            }
            this.flips = newFlips;
        } catch (Exception e) {
            this.errorMessage = "Parse error: " + e.getMessage();
            e.printStackTrace();
        }
    }
    
    private void parseInvestments(String jsonResponse) {
        try {
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(jsonResponse).getAsJsonObject();
            JsonArray investArray = root.getAsJsonArray("investments");
            
            List<RecommendationEntry> newInvest = new ArrayList<RecommendationEntry>();
            for (int i = 0; i < investArray.size(); i++) {
                JsonObject invest = investArray.get(i).getAsJsonObject();
                RecommendationEntry entry = new RecommendationEntry();
                entry.itemId = invest.get("item_id").getAsString();
                entry.currentPrice = invest.get("current_price").getAsDouble();
                entry.predictedPrice = invest.get("predicted_price").getAsDouble();
                entry.expectedChangePct = invest.get("expected_change_pct").getAsDouble();
                entry.confidence = invest.get("confidence").getAsDouble();
                entry.weightedReturn = invest.get("weighted_return").getAsDouble();
                entry.timeframeDays = invest.get("timeframe_days").getAsInt();
                newInvest.add(entry);
            }
            this.investments = newInvest;
        } catch (Exception e) {
            this.errorMessage = "Parse error: " + e.getMessage();
            e.printStackTrace();
        }
    }
    
    private void parseCrashWatch(String jsonResponse) {
        try {
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(jsonResponse).getAsJsonObject();
            JsonArray crashArray = root.getAsJsonArray("crash_items");
            
            List<RecommendationEntry> newCrash = new ArrayList<RecommendationEntry>();
            for (int i = 0; i < crashArray.size(); i++) {
                JsonObject crash = crashArray.get(i).getAsJsonObject();
                RecommendationEntry entry = new RecommendationEntry();
                entry.itemId = crash.get("item_id").getAsString();
                entry.currentPrice = crash.get("current_price").getAsDouble();
                entry.predictedPrice = crash.get("predicted_price").getAsDouble();
                entry.crashPct = crash.get("crash_pct").getAsDouble();
                entry.confidence = crash.get("confidence").getAsDouble();
                entry.crashSeverity = crash.get("crash_severity").getAsDouble();
                entry.reversalHours = crash.get("estimated_reversal_hours").getAsInt();
                entry.recommendation = crash.get("recommendation").getAsString();
                newCrash.add(entry);
            }
            this.crashWatch = newCrash;
        } catch (Exception e) {
            this.errorMessage = "Parse error: " + e.getMessage();
            e.printStackTrace();
        }
    }
    
    private void parseSearchResult(String jsonResponse) {
        try {
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(jsonResponse).getAsJsonObject();

            this.currentItemId = root.has("item_id") ? root.get("item_id").getAsString() : null;
            this.currentEntries.clear();

            if (root.has("entries") && root.get("entries").isJsonArray()) {
                JsonArray arr = root.getAsJsonArray("entries");
                for (int i = 0; i < arr.size(); i++) {
                    JsonObject e = arr.get(i).getAsJsonObject();
                    EntryRecommendation rec = new EntryRecommendation();
                    rec.itemId = this.currentItemId;
                    rec.timestamp = e.has("timestamp") ? e.get("timestamp").getAsString() : "";
                    rec.buyPrice = e.has("buy_price") ? e.get("buy_price").getAsDouble() : 0.0;
                    rec.sellPrice = e.has("sell_price") ? e.get("sell_price").getAsDouble() : 0.0;
                    rec.entryScore = e.has("entry_score") ? e.get("entry_score").getAsDouble() : 0.0;
                    this.currentEntries.add(rec);
                }
            }

            this.scrollOffset = 0;
        } catch (Exception e) {
            this.errorMessage = "Parse error: " + e.getMessage();
            e.printStackTrace();
        }
    }
    
    // Background polling for favorites
    private void startPolling() {
        if (pollingThread != null && pollingThread.isAlive()) {
            return;
        }
        
        shouldPoll = true;
        pollingThread = new Thread(new Runnable() {
            public void run() {
                while (shouldPoll) {
                    try {
                        if (activeInstance != null && !activeInstance.favoriteItems.isEmpty()) {
                            updateFavoritePredictions();
                        }
                        Thread.sleep(10000); // 10 seconds
                    } catch (InterruptedException e) {
                        break;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        pollingThread.setDaemon(true);
        pollingThread.start();
    }
    
    private void stopPolling() {
        shouldPoll = false;
    }
    
    private void updateFavoritePredictions() {
        try {
            List<String> favList = new ArrayList<String>(favoriteItems);
            String itemIds = String.join(",", favList);
            String response = fetchFromUrl(API_URL + "/predictions?item_ids=" + itemIds + "&limit=100");
            
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(response).getAsJsonObject();
            JsonArray predictions = root.getAsJsonArray("predictions");
            
            List<RecommendationEntry> newFavs = new ArrayList<RecommendationEntry>();
            for (int i = 0; i < predictions.size(); i++) {
                JsonObject pred = predictions.get(i).getAsJsonObject();
                RecommendationEntry entry = new RecommendationEntry();
                entry.itemId = pred.get("item_id").getAsString();
                
                // Parse legacy fields
                entry.action = pred.has("action") ? pred.get("action").getAsString() : "BUY";
                entry.currentPrice = pred.has("current_price") ? pred.get("current_price").getAsDouble() : 0;
                entry.predictedPrice = pred.has("predicted_price") ? pred.get("predicted_price").getAsDouble() : 0;
                entry.profitPct = pred.has("expected_profit_pct") ? pred.get("expected_profit_pct").getAsDouble() : 0;
                entry.confidence = pred.has("confidence") ? pred.get("confidence").getAsDouble() : 0;
                
                // Parse three-model fields if available
                if (pred.has("buy_current")) {
                    entry.buyCurrentPrice = pred.get("buy_current").getAsDouble();
                    entry.buyPredictedPrice = pred.get("buy_predicted").getAsDouble();
                    entry.buyChangePct = pred.get("buy_change_pct").getAsDouble();
                    entry.buyDirection = pred.get("buy_direction").getAsString();
                    entry.buyConfidence = pred.get("buy_confidence").getAsDouble();
                    
                    entry.sellCurrentPrice = pred.get("sell_current").getAsDouble();
                    entry.sellPredictedPrice = pred.get("sell_predicted").getAsDouble();
                    entry.sellChangePct = pred.get("sell_change_pct").getAsDouble();
                    entry.sellDirection = pred.get("sell_direction").getAsString();
                    entry.sellConfidence = pred.get("sell_confidence").getAsDouble();
                    
                    entry.spreadCurrent = pred.get("spread_current").getAsDouble();
                    entry.spreadPredicted = pred.get("spread_predicted").getAsDouble();
                    entry.spreadDirection = pred.get("spread_direction").getAsString();
                    entry.spreadConfidence = pred.get("spread_confidence").getAsDouble();
                    
                    entry.flipProfitCurrent = pred.get("flip_profit_current").getAsDouble();
                    entry.flipProfitPredicted = pred.get("flip_profit_predicted").getAsDouble();
                    
                    entry.recommendation = pred.get("recommendation").getAsString();
                }
                
                newFavs.add(entry);
            }
            
            favoritePredictions = newFavs;
        } catch (Exception e) {
            // Silent fail for polling
        }
    }
    
    @Override
    public boolean doesGuiPauseGame() {
        return true;
    }
    
    private void fetchHomeRankings() {
        this.isLoading = true;
        this.errorMessage = null;

        new Thread(new Runnable() {
            public void run() {
                try {
                    String response = fetchFromUrl(API_URL + "/predictions?limit=100&min_score=0.0");
                    parseHomeRankings(response);
                    isLoading = false;
                } catch (Exception e) {
                    errorMessage = "Failed to load homescreen rankings: " + e.getMessage();
                    isLoading = false;
                }
            }
        }).start();
    }
    
    private void parseHomeRankings(String jsonResponse) {
        try {
            JsonParser parser = new JsonParser();
            JsonObject root = parser.parse(jsonResponse).getAsJsonObject();
            this.homeRankings.clear();

            if (root.has("items") && root.get("items").isJsonArray()) {
                JsonArray arr = root.getAsJsonArray("items");
                for (int i = 0; i < arr.size(); i++) {
                    JsonObject obj = arr.get(i).getAsJsonObject();
                    HomeEntrySummary h = new HomeEntrySummary();
                    h.itemId = obj.has("item_id") ? obj.get("item_id").getAsString() : "";
                    h.timestamp = obj.has("timestamp") ? obj.get("timestamp").getAsString() : "";
                    h.buyPrice = obj.has("buy_price") ? obj.get("buy_price").getAsDouble() : 0.0;
                    h.sellPrice = obj.has("sell_price") ? obj.get("sell_price").getAsDouble() : 0.0;
                    h.entryScore = obj.has("entry_score") ? obj.get("entry_score").getAsDouble() : 0.0;
                    h.deltaMinutes = obj.has("delta_minutes") ? obj.get("delta_minutes").getAsDouble() : 0.0;
                    this.homeRankings.add(h);
                }
            }
            this.scrollOffset = 0;
            this.currentPage = 0;
        } catch (Exception e) {
            this.errorMessage = "Parse error (homescreen): " + e.getMessage();
            e.printStackTrace();
        }
    }
    
    private void drawHomeRankings(int panelX, int panelY, int panelWidth, int panelHeight) {
        int headerY = panelY + 50;
        int rowHeight = 36; // Slightly smaller rows to keep bottom text on-screen

        int colRankX = panelX + 15;
        int colItemX = panelX + 60;
        int colTimeX = panelX + 260;
        int colScoreX = panelX + 420;

        drawString(this.fontRendererObj, "#", colRankX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ITEM", colItemX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "ENTRY TIME", colTimeX, headerY, ACCENT_COLOR);
        drawString(this.fontRendererObj, "SCORE", colScoreX, headerY, ACCENT_COLOR);

        drawRect(panelX + 5, headerY + 12, panelX + panelWidth - 5, headerY + 13, BORDER_COLOR);

        int startY = headerY + 20;
        int total = homeRankings != null ? homeRankings.size() : 0;
        int startIndex = currentPage * ROWS_PER_PAGE;
        int endIndex = Math.min(startIndex + ROWS_PER_PAGE, total);
        int visibleItems = Math.max(0, endIndex - startIndex);

        for (int i = 0; i < visibleItems; i++) {
            int index = startIndex + i;
            if (index >= total) break;

            HomeEntrySummary h = homeRankings.get(index);
            int rowY = startY + i * rowHeight;

            if (i % 2 == 0) {
                drawRect(panelX + 5, rowY - 2, panelX + panelWidth - 5, rowY + rowHeight - 7, 0x40000000);
            }

            drawString(this.fontRendererObj, "#" + (index + 1), colRankX, rowY + 5, TEXT_PRIMARY);

            String itemName = (h.itemId != null ? h.itemId : "").replace("_", " ");
            if (itemName.length() > 22) itemName = itemName.substring(0, 19) + "...";
            drawString(this.fontRendererObj, itemName, colItemX, rowY + 5, TEXT_PRIMARY);

            String timeStr = h.timestamp != null ? h.timestamp : "";
            if (timeStr.length() > 19) timeStr = timeStr.substring(0, 19);
            drawString(this.fontRendererObj, timeStr, colTimeX, rowY + 5, TEXT_SECONDARY);

            String scoreStr = String.format("%.4f", h.entryScore);
            drawString(this.fontRendererObj, scoreStr, colScoreX, rowY + 5, ACCENT_COLOR);
        }

        if (total > ROWS_PER_PAGE) {
            int totalPages = (total + ROWS_PER_PAGE - 1) / ROWS_PER_PAGE;
            String pageText = String.format("Page %d/%d", currentPage + 1, totalPages);
            int pageX = panelX + panelWidth - this.fontRendererObj.getStringWidth(pageText) - 10;
            int pageY = panelY + panelHeight - 20;
            drawString(this.fontRendererObj, pageText, pageX, pageY, TEXT_SECONDARY);
        }
    }
    
    // Simple entry representation from the new Flask API
    private static class EntryRecommendation {
        String itemId;
        String timestamp;
        double buyPrice;
        double sellPrice;
        double entryScore;
    }
    
    // Aggregated per-item summary for homescreen ranking
    private static class HomeEntrySummary {
        String itemId;
        String timestamp;
        double buyPrice;
        double sellPrice;
        double entryScore;
        double deltaMinutes;
    }
    
    private static class RecommendationEntry {
        String itemId;
        String action;  // Legacy: BUY or SELL based on buy price
        double currentPrice;  // Legacy: buy price
        double predictedPrice;  // Legacy: buy price predicted
        double profitPct;  // Legacy: buy price change
        double confidence;  // Legacy: buy price confidence
        
        // NEW: Comprehensive three-model predictions
        // Buy price
        double buyCurrentPrice;
        double buyPredictedPrice;
        double buyChangePct;
        String buyDirection;
        double buyConfidence;
        
        // Sell price
        double sellCurrentPrice;
        double sellPredictedPrice;
        double sellChangePct;
        String sellDirection;
        double sellConfidence;
        
        // Spread
        double spreadCurrent;
        double spreadPredicted;
        String spreadDirection;
        double spreadConfidence;
        
        // Flip profit
        double flipProfitCurrent;
        double flipProfitPredicted;
        
        // Smart recommendation
        String recommendation;  // STRONG_BUY, BUY, SELL, STRONG_SELL, WAIT, ARBITRAGE, HOLD
        
        // Flips-specific fields
        double buyOrderPrice;   // Price to pay for buy order
        double sellOrderPrice;  // Price to get from sell order  
        double spreadValue;     // Absolute spread value
        double spreadPct;       // Spread as percentage
        
        // Investments-specific fields
        double expectedChangePct;  // Expected price change %
        double weightedReturn;     // Confidence * expected change
        int timeframeDays;         // Investment timeframe (1, 7, or 30)
        
        // Crash Watch-specific fields
        double crashPct;          // Crash percentage
        double crashSeverity;     // Severity metric
        int reversalHours;        // Estimated hours until reversal
    }
}
