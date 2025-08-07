import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'services/url_monitor_service.dart';
import 'services/system_browser_monitor.dart';
import 'services/accessibility_url_monitor.dart';
import 'services/network_monitor_service.dart';
import 'services/clipboard_monitor_service.dart';
import 'services/background_url_service.dart';
import 'screens/url_history_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final urlService = UrlMonitorService();
  await urlService.initialize();

  final systemBrowserMonitor = SystemBrowserMonitor();
  await systemBrowserMonitor.initialize();

  final accessibilityMonitor = AccessibilityUrlMonitor();
  accessibilityMonitor.startMonitoring();

  final networkMonitor = NetworkMonitorService();
  await networkMonitor.initialize();

  final clipboardMonitor = ClipboardMonitorService();
  clipboardMonitor.startMonitoring();

  final backgroundUrlService = BackgroundUrlService();
  await backgroundUrlService.initialize();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebView Scraper',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const WebViewScreen(),
    );
  }
}

class WebViewScreen extends StatefulWidget {
  const WebViewScreen({super.key});

  @override
  State<WebViewScreen> createState() => _WebViewScreenState();
}

class _WebViewScreenState extends State<WebViewScreen> {
  late final WebViewController _controller;
  final TextEditingController _urlController = TextEditingController();
  final TextEditingController _selectorController = TextEditingController();
  final UrlMonitorService _urlService = UrlMonitorService();
  final SystemBrowserMonitor _systemBrowserMonitor = SystemBrowserMonitor();
  final AccessibilityUrlMonitor _accessibilityMonitor =
      AccessibilityUrlMonitor();
  final NetworkMonitorService _networkMonitor = NetworkMonitorService();
  final ClipboardMonitorService _clipboardMonitor = ClipboardMonitorService();
  final BackgroundUrlService _backgroundUrlService = BackgroundUrlService();
  String _scrapedContent = '';
  bool _isLoading = false;
  String _currentUrl = '';
  int _urlLogCount = 0;
  bool _accessibilityServiceEnabled = false;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (String url) {
            setState(() {
              _isLoading = true;
              _currentUrl = url;
              _urlController.text = url;
            });
            _logUrl(url);
          },
          onPageFinished: (String url) {
            setState(() {
              _isLoading = false;
              _currentUrl = url;
              _urlController.text = url;
            });
            _logUrl(url);
          },
          onUrlChange: (UrlChange change) {
            if (change.url != null) {
              _logUrl(change.url!);
              setState(() {
                _currentUrl = change.url!;
                _urlController.text = change.url!;
              });
            }
          },
        ),
      )
      ..loadRequest(Uri.parse('https://example.com'));

    _urlService.urlStream.listen((_) {
      if (mounted) {
        setState(() {
          _urlLogCount = _urlService.urlHistory.length;
        });
      }
    });

    _urlLogCount = _urlService.urlHistory.length;
    _checkAccessibilityServiceStatus();
  }

  Future<void> _checkAccessibilityServiceStatus() async {
    final isEnabled = await _backgroundUrlService
        .isAccessibilityServiceEnabled();
    setState(() {
      _accessibilityServiceEnabled = isEnabled;
    });
  }

  void _logUrl(String url) {
    _urlService.logUrl(url, 'WebView');
  }

  void _loadUrl() {
    String url = _urlController.text.trim();
    if (url.isNotEmpty) {
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://$url';
      }
      _controller.loadRequest(Uri.parse(url));
      _logUrl(url);
    }
  }

  Future<void> _scrapeContent() async {
    String selector = _selectorController.text.trim();
    if (selector.isEmpty) {
      selector = 'title, h1, h2, h3, p';
    }

    try {
      final result = await _controller.runJavaScriptReturningResult('''
        (function() {
          const elements = document.querySelectorAll('$selector');
          const content = [];
          elements.forEach(function(element, index) {
            if (index < 20) {
              content.push({
                tag: element.tagName.toLowerCase(),
                text: element.textContent.trim(),
                innerHTML: element.innerHTML
              });
            }
          });
          return JSON.stringify(content);
        })();
      ''');

      setState(() {
        _scrapedContent = result.toString();
      });
    } catch (e) {
      setState(() {
        _scrapedContent = 'Error scraping content: $e';
      });
    }
  }

  Future<void> _scrapeLinks() async {
    try {
      final result = await _controller.runJavaScriptReturningResult('''
        (function() {
          const links = document.querySelectorAll('a[href]');
          const linkData = [];
          links.forEach(function(link, index) {
            if (index < 10) {
              linkData.push({
                text: link.textContent.trim(),
                href: link.href,
                title: link.title || ''
              });
            }
          });
          return JSON.stringify(linkData);
        })();
      ''');

      setState(() {
        _scrapedContent = result.toString();
      });
    } catch (e) {
      setState(() {
        _scrapedContent = 'Error scraping links: $e';
      });
    }
  }

  void _openUrlHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const UrlHistoryScreen()),
    );
  }

  void _testUrlCapture() async {
    await _backgroundUrlService.testBackgroundService();
    await _clipboardMonitor.checkClipboardNow();
    await _systemBrowserMonitor.simulateBrowserActivity(
      'https://www.realme.com/th/',
    );
    _systemBrowserMonitor.simulateTyping('realme.com');
    _systemBrowserMonitor.simulateTyping(
      'https://developer.android.com/guide/topics/manifest/service-element?hl=th',
    );
    await _systemBrowserMonitor.detectClipboardFromBrowser();
    await _clipboardMonitor.simulateUrlDetection('https://www.realme.com/th/');
    await _networkMonitor.captureManualUrl(
      'https://developer.android.com/guide/topics/manifest/service-element?hl=th',
      'Manual Test: Android Developer',
    );
    await _networkMonitor.captureManualUrl(
      'https://www.google.com/search?q=mobile+service+android',
      'Manual Test: Google Search',
    );
    await _checkAccessibilityServiceStatus();

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Background monitoring test:\n'
            '• Accessibility Service: ${_accessibilityServiceEnabled ? "Enabled" : "Disabled"}\n'
            '• Background Listening: ${_backgroundUrlService.isListening ? "Active" : "Inactive"}\n'
            '• Service Connected: ${_backgroundUrlService.isServiceConnected ? "Yes" : "No"}\n'
            '• Browser Active: ${_systemBrowserMonitor.browserIsActive}\n'
            '• URLs Detected: ${_systemBrowserMonitor.detectedUrlsCount}',
          ),
          duration: Duration(seconds: 6),
          action: !_accessibilityServiceEnabled
              ? SnackBarAction(
                  label: 'Enable Service',
                  onPressed: () =>
                      _backgroundUrlService.openAccessibilitySettings(),
                )
              : null,
        ),
      );
    }
  }

  Widget _buildMonitoringStatus() {
    Color statusColor;
    String statusText;

    if (_backgroundUrlService.isServiceConnected) {
      statusColor = Colors.green[50]!;
      statusText = 'Background: Active';
    } else if (_systemBrowserMonitor.browserIsActive) {
      statusColor = Colors.blue[50]!;
      statusText =
          'Browser: Active (${_systemBrowserMonitor.detectedUrlsCount} URLs)';
    } else {
      statusColor = Colors.orange[50]!;
      statusText =
          'Network: ${_networkMonitor.isMonitoring ? "Active" : "Inactive"}';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: statusColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: statusColor.withOpacity(0.5)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(width: 4),
          Text(
            statusText,
            style: TextStyle(
              fontSize: 11,
              color: statusColor.withOpacity(1.0),
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildClipboardStatus() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: _accessibilityServiceEnabled ? Colors.green[50] : Colors.red[50],
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _accessibilityServiceEnabled
              ? Colors.green[200]!
              : Colors.red[200]!,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(width: 4),
          Text(
            _accessibilityServiceEnabled
                ? 'Accessibility: ON'
                : 'Accessibility: OFF',
            style: TextStyle(
              fontSize: 11,
              color: _accessibilityServiceEnabled
                  ? Colors.green[700]
                  : Colors.red[700],
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('WebView Scraper'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (!_accessibilityServiceEnabled)
            IconButton(
              onPressed: () async {
                await _backgroundUrlService.openAccessibilitySettings();
                await Future.delayed(Duration(seconds: 1));
                await _checkAccessibilityServiceStatus();
              },
              icon: const Text('Settings'),
              tooltip: 'Enable Background Monitoring',
            ),
          IconButton(
            onPressed: _testUrlCapture,
            icon: const Text('Test'),
            tooltip: 'Test URL Capture',
          ),
          Stack(
            children: [
              IconButton(
                onPressed: _openUrlHistory,
                icon: const Text('History'),
              ),
              if (_urlLogCount > 0)
                Positioned(
                  right: 8,
                  top: 8,
                  child: Container(
                    padding: const EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    constraints: const BoxConstraints(
                      minWidth: 16,
                      minHeight: 16,
                    ),
                    child: Text(
                      _urlLogCount > 99 ? '99+' : _urlLogCount.toString(),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(8.0),
            color: Colors.grey[100],
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _urlController,
                        decoration: const InputDecoration(
                          hintText: 'Enter URL...',
                          border: OutlineInputBorder(),
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                        ),
                        onSubmitted: (_) => _loadUrl(),
                        onChanged: (value) {
                          _accessibilityMonitor.simulateKeyboardTyping(value);
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    ElevatedButton(
                      onPressed: _loadUrl,
                      child: const Text('Go'),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    IconButton(
                      onPressed: () => _controller.goBack(),
                      icon: const Text('Back'),
                    ),
                    IconButton(
                      onPressed: () => _controller.goForward(),
                      icon: const Text('Forward'),
                    ),
                    IconButton(
                      onPressed: () => _controller.reload(),
                      icon: const Text('Reload'),
                    ),
                    const SizedBox(width: 8),
                    _buildMonitoringStatus(),
                    const SizedBox(width: 4),
                    _buildClipboardStatus(),
                    const Spacer(),
                    StreamBuilder<UrlLogEntry>(
                      stream: _urlService.urlStream,
                      builder: (context, snapshot) {
                        return Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: snapshot.hasData
                                ? Colors.green
                                : Colors.orange,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const SizedBox(width: 4),
                              Text(
                                snapshot.hasData ? 'Logging' : 'Monitoring',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                    if (_isLoading) ...[
                      const SizedBox(width: 8),
                      const CircularProgressIndicator(),
                    ],
                  ],
                ),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _selectorController,
                        decoration: const InputDecoration(
                          hintText: 'CSS selector (e.g., h1, .class, #id)',
                          border: OutlineInputBorder(),
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    ElevatedButton(
                      onPressed: _scrapeContent,
                      child: const Text('Scrape'),
                    ),
                    const SizedBox(width: 4),
                    ElevatedButton(
                      onPressed: _scrapeLinks,
                      child: const Text('Links'),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Expanded(flex: 2, child: WebViewWidget(controller: _controller)),
          if (_scrapedContent.isNotEmpty)
            Expanded(
              flex: 1,
              child: Container(
                width: double.infinity,
                color: Colors.grey[50],
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(8.0),
                      color: Colors.blue[100],
                      child: Row(
                        children: [
                          const Text(
                            'Scraped Content:',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          const Spacer(),
                          IconButton(
                            onPressed: () =>
                                setState(() => _scrapedContent = ''),
                            icon: const Text('Close'),
                          ),
                        ],
                      ),
                    ),
                    Expanded(
                      child: SingleChildScrollView(
                        padding: const EdgeInsets.all(8.0),
                        child: SelectableText(
                          _formatScrapedContent(_scrapedContent),
                          style: const TextStyle(
                            fontSize: 12,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  String _formatScrapedContent(String content) {
    try {
      if (content.startsWith('"') && content.endsWith('"')) {
        content = content.substring(1, content.length - 1);
      }

      content = content.replaceAll('\\n', '\n');
      content = content.replaceAll('\\"', '"');
      content = content.replaceAll('\\/', '/');

      return content;
    } catch (e) {
      return content;
    }
  }

  @override
  void dispose() {
    _urlController.dispose();
    _selectorController.dispose();
    super.dispose();
  }
}
