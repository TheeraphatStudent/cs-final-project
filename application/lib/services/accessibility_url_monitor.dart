import 'dart:async';
import 'dart:math';
import 'url_monitor_service.dart';

class AccessibilityUrlMonitor {
  static final AccessibilityUrlMonitor _instance =
      AccessibilityUrlMonitor._internal();
  factory AccessibilityUrlMonitor() => _instance;
  AccessibilityUrlMonitor._internal();

  final UrlMonitorService _urlService = UrlMonitorService();
  bool _isMonitoring = false;
  final Random _random = Random();

  final List<String> _searchQueries = [
    'mobile service',
    'android development',
    'mobile app service',
    'device management',
    'system service',
    'background service',
    'mobile security',
    'android service',
    'device monitoring',
    'system monitoring',
  ];

  void startMonitoring() {
    if (_isMonitoring) return;

    _isMonitoring = true;
  }

  String _getRandomBrowser() {
    List<String> browsers = [
      'Chrome',
      'Firefox',
      'Edge',
      'Opera',
      'Samsung Browser',
    ];
    return browsers[_random.nextInt(browsers.length)];
  }

  void addDetectedUrl(String url, String source) {
    _urlService.logUrl(url, 'Detected: $source');
  }

  void simulateKeyboardTyping(String text) {
    if (text.contains('http') ||
        text.contains('www.') ||
        text.contains('.com')) {
      _urlService.logUrl(text, 'Keyboard Input');
    }
  }

  bool get isMonitoring => _isMonitoring;

  void stopMonitoring() {
    _isMonitoring = false;
  }

  void dispose() {
    stopMonitoring();
  }
}
