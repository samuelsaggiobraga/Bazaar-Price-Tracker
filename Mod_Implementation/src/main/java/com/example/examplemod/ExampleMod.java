package com.example.examplemod;

import net.minecraft.client.Minecraft;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.EnumChatFormatting;
import net.minecraftforge.client.ClientCommandHandler;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

@Mod(modid = ExampleMod.MODID, version = ExampleMod.VERSION, name = "Bazaar Investment Mod")
public class ExampleMod
{
    public static final String MODID = "bazaarmod";
    public static final String VERSION = "1.0";
    private static final String API_URL = "http://localhost:5001/recommendations";
    
    @EventHandler
    public void init(FMLInitializationEvent event)
    {
        // Register the commands
        ClientCommandHandler.instance.registerCommand(new BazaarCommand());
        ClientCommandHandler.instance.registerCommand(new BazaarPredictCommand());
        ClientCommandHandler.instance.registerCommand(new BazaarGUICommand());
        System.out.println("[Bazaar Mod] Initialized!");
        System.out.println("[Bazaar Mod] Use /bazaar or /bz to get investment recommendations.");
        System.out.println("[Bazaar Mod] Use /bzpredict <item_id> to get prediction using client-side API calls.");
        System.out.println("[Bazaar Mod] Use /bzgui to open the Bazaar Tracker GUI.");
    }
    
    public static class BazaarCommand extends CommandBase
    {
        @Override
        public String getCommandName() {
            return "bazaar";
        }
        
        @Override
        public String getCommandUsage(ICommandSender sender) {
            return "/bazaar [limit] - Get bazaar investment recommendations";
        }
        
        @Override
        public java.util.List<String> getCommandAliases() {
            java.util.List<String> aliases = new java.util.ArrayList<String>();
            aliases.add("bz");
            return aliases;
        }
        
        @Override
        public int getRequiredPermissionLevel() {
            return 0; // No permission required
        }
        
        @Override
        public void processCommand(ICommandSender sender, String[] args) throws CommandException {
            int limit = 5;
            
            // Parse limit argument
            if (args.length > 0) {
                try {
                    limit = Integer.parseInt(args[0]);
                    if (limit < 1 || limit > 20) {
                        sender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.RED + "Limit must be between 1 and 20"
                        ));
                        return;
                    }
                } catch (NumberFormatException e) {
                    sender.addChatMessage(new ChatComponentText(
                        EnumChatFormatting.RED + "Invalid number: " + args[0]
                    ));
                    return;
                }
            }
            
            final int finalLimit = limit;
            
            sender.addChatMessage(new ChatComponentText(
                EnumChatFormatting.YELLOW + "Fetching bazaar recommendations..."
            ));
            
            // Fetch in a separate thread to avoid freezing the game
            final ICommandSender finalSender = sender;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        String response = fetchRecommendations(finalLimit);
                        displayRecommendations(finalSender, response);
                    } catch (Exception e) {
                        finalSender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.RED + "Error: " + e.getMessage()
                        ));
                        finalSender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.GRAY + "Make sure the Flask server is running on port 5001"
                        ));
                    }
                }
            }).start();
        }
        
        private String fetchRecommendations(int limit) throws Exception {
            URL url = new URL(API_URL + "?limit=" + limit + "&min_confidence=60");
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
        
        private void displayRecommendations(ICommandSender sender, String jsonResponse) {
            try {
                JsonParser parser = new JsonParser();
                JsonObject root = parser.parse(jsonResponse).getAsJsonObject();
                JsonArray recommendations = root.getAsJsonArray("recommendations");
                
                if (recommendations.size() == 0) {
                    sender.addChatMessage(new ChatComponentText(
                        EnumChatFormatting.YELLOW + "No high-confidence recommendations found."
                    ));
                    return;
                }
                
                // Header
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.GOLD + "" + EnumChatFormatting.BOLD + "=== Bazaar Investment Recommendations ==="
                ));
                
                // Display each recommendation
                for (int i = 0; i < recommendations.size(); i++) {
                    JsonObject rec = recommendations.get(i).getAsJsonObject();
                    
                    String itemId = rec.get("item_id").getAsString();
                    String action = rec.get("action").getAsString();
                    double currentPrice = rec.get("current_price").getAsDouble();
                    double predictedPrice = rec.get("predicted_price").getAsDouble();
                    double profitPct = rec.get("expected_profit_pct").getAsDouble();
                    double confidence = rec.get("confidence").getAsDouble();
                    
                    // Color based on action
                    EnumChatFormatting actionColor = action.equals("BUY") ? 
                        EnumChatFormatting.GREEN : EnumChatFormatting.RED;
                    
                    // Format item name (remove underscores)
                    String itemName = itemId.replace("_", " ");
                    
                    sender.addChatMessage(new ChatComponentText(
                        EnumChatFormatting.AQUA + "" + (i + 1) + ". " +
                        EnumChatFormatting.WHITE + itemName
                    ));
                    
                    sender.addChatMessage(new ChatComponentText(
                        "   " + actionColor + "" + EnumChatFormatting.BOLD + action +
                        EnumChatFormatting.GRAY + " | Profit: " +
                        EnumChatFormatting.GOLD + String.format("%.1f%%", profitPct) +
                        EnumChatFormatting.GRAY + " | Confidence: " +
                        EnumChatFormatting.YELLOW + String.format("%.0f%%", confidence)
                    ));
                    
                    sender.addChatMessage(new ChatComponentText(
                        EnumChatFormatting.GRAY + "   Current: " +
                        EnumChatFormatting.WHITE + String.format("%.2f", currentPrice) +
                        EnumChatFormatting.GRAY + " → Predicted: " +
                        EnumChatFormatting.WHITE + String.format("%.2f", predictedPrice)
                    ));
                }
                
                // Footer
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.GRAY + "Showing " + recommendations.size() + " recommendations"
                ));
                
            } catch (Exception e) {
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.RED + "Error parsing response: " + e.getMessage()
                ));
                e.printStackTrace();
            }
        }
    }
    
    public static class BazaarPredictCommand extends CommandBase
    {
        private static final String COFLNET_API = "https://sky.coflnet.com/api/bazaar";
        private static final String SERVER_API = "http://localhost:5001/predict/with-data";
        
        @Override
        public String getCommandName() {
            return "bzpredict";
        }
        
        @Override
        public String getCommandUsage(ICommandSender sender) {
            return "/bzpredict <item_id> - Get price prediction using client-side API calls";
        }
        
        @Override
        public java.util.List<String> getCommandAliases() {
            java.util.List<String> aliases = new java.util.ArrayList<String>();
            aliases.add("bzpred");
            return aliases;
        }
        
        @Override
        public int getRequiredPermissionLevel() {
            return 0;
        }
        
        @Override
        public void processCommand(ICommandSender sender, String[] args) throws CommandException {
            if (args.length == 0) {
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.RED + "Usage: /bzpredict <item_id>"
                ));
                return;
            }
            
            final String itemId = args[0];
            final ICommandSender finalSender = sender;
            
            sender.addChatMessage(new ChatComponentText(
                EnumChatFormatting.YELLOW + "Fetching data for " + itemId + "..."
            ));
            
            // Fetch in a separate thread to avoid freezing the game
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        // Step 1: Fetch raw data from Coflnet API (client-side)
                        String apiDataUrl = COFLNET_API + "/" + itemId + "/history/day";
                        String apiDataJson = fetchFromUrl(apiDataUrl);
                        JsonArray apiData = new JsonParser().parse(apiDataJson).getAsJsonArray();
                        
                        if (apiData.size() == 0) {
                            finalSender.addChatMessage(new ChatComponentText(
                                EnumChatFormatting.RED + "No data available for item: " + itemId
                            ));
                            return;
                        }
                        
                        finalSender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.GREEN + "✓ Fetched " + apiData.size() + " data points"
                        ));
                        
                        // Step 2: Send data to server for prediction
                        JsonObject requestBody = new JsonObject();
                        requestBody.addProperty("item_id", itemId);
                        requestBody.add("api_data", apiData);
                        
                        String response = postToServer(SERVER_API, requestBody.toString());
                        JsonObject prediction = new JsonParser().parse(response).getAsJsonObject();
                        
                        // Step 3: Display prediction
                        displayPrediction(finalSender, prediction);
                        
                    } catch (Exception e) {
                        finalSender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.RED + "Error: " + e.getMessage()
                        ));
                        finalSender.addChatMessage(new ChatComponentText(
                            EnumChatFormatting.GRAY + "Make sure the Flask server is running on port 5001"
                        ));
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
                throw new Exception("Coflnet returned code: " + responseCode);
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
        
        private String postToServer(String urlString, String jsonData) throws Exception {
            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);
            conn.setDoOutput(true);
            
            // Send POST data
            OutputStream os = conn.getOutputStream();
            byte[] input = jsonData.getBytes("utf-8");
            os.write(input, 0, input.length);
            os.close();
            
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
        
        private void displayPrediction(ICommandSender sender, JsonObject prediction) {
            try {
                String itemId = prediction.get("item_id").getAsString();
                String action = prediction.get("recommendation").getAsString();
                double currentPrice = prediction.get("current_price").getAsDouble();
                double predictedPrice = prediction.get("predicted_price").getAsDouble();
                double profitPct = prediction.get("predicted_change_pct").getAsDouble();
                double confidence = prediction.get("confidence").getAsDouble();
                String direction = prediction.get("direction").getAsString();
                
                // Color based on action
                EnumChatFormatting actionColor = action.equals("BUY") ? 
                    EnumChatFormatting.GREEN : (action.equals("SELL") ? EnumChatFormatting.RED : EnumChatFormatting.YELLOW);
                
                // Format item name
                String itemName = itemId.replace("_", " ");
                
                // Header
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.GOLD + "" + EnumChatFormatting.BOLD + "=== Prediction Result ==="
                ));
                
                // Item name
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.AQUA + "Item: " + EnumChatFormatting.WHITE + itemName
                ));
                
                // Action
                sender.addChatMessage(new ChatComponentText(
                    "   " + actionColor + "" + EnumChatFormatting.BOLD + action +
                    EnumChatFormatting.GRAY + " | Direction: " +
                    (direction.equals("UP") ? EnumChatFormatting.GREEN : EnumChatFormatting.RED) + direction
                ));
                
                // Prices
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.GRAY + "   Current: " +
                    EnumChatFormatting.WHITE + String.format("%.2f", currentPrice) +
                    EnumChatFormatting.GRAY + " → Predicted: " +
                    EnumChatFormatting.WHITE + String.format("%.2f", predictedPrice)
                ));
                
                // Profit
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.GRAY + "   Profit: " +
                    EnumChatFormatting.GOLD + String.format("%.2f%%", profitPct) +
                    EnumChatFormatting.GRAY + " | Confidence: " +
                    EnumChatFormatting.YELLOW + String.format("%.0f%%", confidence)
                ));
                
            } catch (Exception e) {
                sender.addChatMessage(new ChatComponentText(
                    EnumChatFormatting.RED + "Error parsing prediction: " + e.getMessage()
                ));
            }
        }
    }
    
    public static class BazaarGUICommand extends CommandBase
    {
        @Override
        public String getCommandName() {
            return "bzgui";
        }
        
        @Override
        public String getCommandUsage(ICommandSender sender) {
            return "/bzgui - Open Bazaar Tracker GUI";
        }
        
        @Override
        public java.util.List<String> getCommandAliases() {
            java.util.List<String> aliases = new java.util.ArrayList<String>();
            aliases.add("bztracker");
            return aliases;
        }
        
        @Override
        public int getRequiredPermissionLevel() {
            return 0;
        }
        
        @Override
        public void processCommand(ICommandSender sender, String[] args) throws CommandException {
            final ICommandSender finalSender = sender;
            sender.addChatMessage(new ChatComponentText(
                EnumChatFormatting.YELLOW + "Opening Bazaar Tracker GUI..."
            ));
            
            sender.addChatMessage(new ChatComponentText(
                EnumChatFormatting.GRAY + "Debug: Command received"
            ));
            
            // Schedule GUI opening AFTER chat closes
            final Minecraft mc = Minecraft.getMinecraft();
            
            // Wait for next tick to ensure chat is closed
            new Thread(new Runnable() {
                public void run() {
                    try {
                        // Wait a bit for chat to close
                        Thread.sleep(100);
                        
                        // Now schedule on main thread
                        mc.addScheduledTask(new Runnable() {
                            public void run() {
                                mc.displayGuiScreen(new BazaarTrackerGUI());
                            }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        }
    }
}
