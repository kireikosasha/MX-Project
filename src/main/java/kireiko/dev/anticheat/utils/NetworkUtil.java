package kireiko.dev.anticheat.utils;

import kireiko.dev.anticheat.MX;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.stream.Collectors;


public class NetworkUtil {
    private static final String apiUrl = "https://paste.mineland.net/documents";
    private static final int CONNECT_TIMEOUT_MS = 5000;
    private static final int READ_TIMEOUT_MS = 10000;

    public static String createPaste(final String textToPaste) {
        final String userAgent = "MX/1.0 (Anti-Cheat service)";
        String pasteLink = null;
        HttpURLConnection connection = null;

        try {
            final URL url = new URL(apiUrl);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("User-Agent", userAgent);
            connection.setDoOutput(true);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            try (DataOutputStream dos = new DataOutputStream(connection.getOutputStream())) {
                dos.writeBytes(textToPaste);
                dos.flush();
            }
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                try (BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()))) {
                    String response = in.lines().collect(Collectors.joining());
                    String[] parts = response.split("\"");
                    if (parts.length > 3) {
                        pasteLink = "https://paste.mineland.net/" + parts[3];
                    } else {
                        MX.getInstance().getLogger().warning("Unexpected paste response: " + response);
                    }
                }
            } else {
                MX.getInstance().getLogger().severe("Error while creating document: " + responseCode);
            }
        } catch (IOException e) {
            MX.getInstance().getLogger().warning("Failed to create paste: " + e.getMessage());
        } finally {
            if (connection != null) connection.disconnect();
        }

        return pasteLink;
    }
}
