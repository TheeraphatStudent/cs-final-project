package com.example.application

import android.accessibilityservice.AccessibilityService
import android.content.ClipboardManager
import android.content.Context
import android.view.accessibility.AccessibilityEvent
import io.flutter.plugin.common.EventChannel
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.engine.dart.DartExecutor
import java.util.regex.Pattern

class UrlClipboardAccessibilityService : AccessibilityService() {
    
    companion object {
        private const val TAG = "UrlClipboardService"
        private const val CHANNEL_NAME = "com.example.application/url_clipboard"
        private var eventSink: EventChannel.EventSink? = null
        private var instance: UrlClipboardAccessibilityService? = null
        
        fun setEventSink(sink: EventChannel.EventSink?) {
            eventSink = sink
        }
        
        fun getInstance(): UrlClipboardAccessibilityService? = instance
    }
    
    private lateinit var clipboardManager: ClipboardManager
    private var lastClipboardContent: String = ""
    
    private val urlPattern = Pattern.compile(
        "^(https?://)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&//=]*)",
        Pattern.CASE_INSENSITIVE
    )
    
    private val clipboardListener = ClipboardManager.OnPrimaryClipChangedListener {
        checkClipboardForUrl()
    }
    
    override fun onCreate() {
        super.onCreate()
        instance = this
        clipboardManager = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    }
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        
        clipboardManager.addPrimaryClipChangedListener(clipboardListener)
        
        sendEventToFlutter(hashMapOf<String, Any>(
            "type" to "service_status",
            "status" to "connected",
            "message" to "Background URL monitoring active"
        ))
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        clipboardManager.removePrimaryClipChangedListener(clipboardListener)
        
        sendEventToFlutter(hashMapOf<String, Any>(
            "type" to "service_status",
            "status" to "disconnected",
            "message" to "Background URL monitoring stopped"
        ))
        
        instance = null
    }
    
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.let {
            when (it.eventType) {
                AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                    handleWindowStateChanged(it)
                }
                AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> {
                    handleTextChanged(it)
                }
            }
        }
    }
    
    override fun onInterrupt() {
    }
    
    private fun checkClipboardForUrl() {
        try {
            val clipData = clipboardManager.primaryClip
            if (clipData != null && clipData.itemCount > 0) {
                val clipText = clipData.getItemAt(0).text?.toString()?.trim()
                
                if (!clipText.isNullOrEmpty() && clipText != lastClipboardContent) {
                    lastClipboardContent = clipText
                    
                    if (isValidUrl(clipText)) {
                        val browserSource = detectBrowserSource(clipText)
                        
                        sendEventToFlutter(hashMapOf<String, Any>(
                            "type" to "url_detected",
                            "url" to clipText,
                            "source" to "Background: $browserSource",
                            "timestamp" to System.currentTimeMillis(),
                            "detection_method" to "clipboard_monitoring"
                        ))
                    }
                }
            }
        } catch (e: Exception) {
        }
    }
    
    private fun handleWindowStateChanged(event: AccessibilityEvent) {
        val packageName = event.packageName?.toString()
        val browserApps = listOf(
            "com.android.chrome",
            "org.mozilla.firefox",
            "com.microsoft.emmx",
            "com.opera.browser",
            "com.sec.android.app.sbrowser"
        )
        
        if (packageName in browserApps) {
            val browserName = getBrowserName(packageName)
            
            sendEventToFlutter(hashMapOf<String, Any>(
                "type" to "browser_activity",
                "browser" to browserName,
                "package" to (packageName ?: "unknown"),
                "timestamp" to System.currentTimeMillis()
            ))
        }
    }
    
    private fun handleTextChanged(event: AccessibilityEvent) {
        val text = event.text?.joinToString(" ")?.trim()
        if (!text.isNullOrEmpty() && isValidUrl(text)) {
            
            sendEventToFlutter(hashMapOf<String, Any>(
                "type" to "url_detected",
                "url" to text,
                "source" to "Background: Browser Typing",
                "timestamp" to System.currentTimeMillis(),
                "detection_method" to "text_change_monitoring"
            ))
        }
    }
    
    private fun isValidUrl(text: String): Boolean {
        if (text.length < 7) return false
        
        return urlPattern.matcher(text).matches() ||
               text.startsWith("http://") ||
               text.startsWith("https://") ||
               text.contains("www.") ||
               text.contains(".com") ||
               text.contains(".org") ||
               text.contains(".net") ||
               text.contains(".th") ||
               text.contains(".dev") ||
               text.contains(".io")
    }
    
    private fun detectBrowserSource(url: String): String {
        return when {
            url.contains("realme.com") -> "Chrome (Shopping)"
            url.contains("developer.android.com") -> "Chrome (Developer)"
            url.contains("google.com/search") -> "Chrome (Search)"
            url.contains("youtube.com") -> "Chrome (Video)"
            url.contains("github.com") -> "Chrome (Code)"
            url.contains("stackoverflow.com") -> "Chrome (Development)"
            url.contains("facebook.com") || url.contains("instagram.com") -> "Chrome (Social)"
            else -> "Browser Copy/Paste"
        }
    }
    
    private fun getBrowserName(packageName: String?): String {
        return when (packageName) {
            "com.android.chrome" -> "Chrome"
            "org.mozilla.firefox" -> "Firefox"
            "com.microsoft.emmx" -> "Edge"
            "com.opera.browser" -> "Opera"
            "com.sec.android.app.sbrowser" -> "Samsung Browser"
            else -> "Unknown Browser"
        }
    }
    
    private fun sendEventToFlutter(data: HashMap<String, Any>) {
        try {
            eventSink?.success(data)
        } catch (e: Exception) {
        }
    }
} 