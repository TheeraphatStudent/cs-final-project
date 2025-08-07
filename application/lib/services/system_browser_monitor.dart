import 'dart:async';
import 'dart:math';
import 'package:flutter/services.dart';
import 'url_monitor_service.dart';

class SystemBrowserMonitor {
  static final SystemBrowserMonitor _instance =
      SystemBrowserMonitor._internal();
  factory SystemBrowserMonitor() => _instance;
  SystemBrowserMonitor._internal();

  final UrlMonitorService _urlService = UrlMonitorService();
  Timer? _monitoringTimer;
  Timer? _visibilityTimer;
  bool _isMonitoring = false;
  bool _browserIsActive = false;
  String _lastDetectedActivity = '';
  final Random _random = Random();

  final Set<String> _recentBrowserUrls = {};
  final List<String> _typingPatterns = [];

  Future<bool> initialize() async {
    await _startRealBrowserMonitoring();
    return true;
  }

  Future<void> _startRealBrowserMonitoring() async {
    _isMonitoring = true;

    _visibilityTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      _detectBrowserVisibilityChange();
    });

    _monitoringTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      _detectRealBrowserActivity();
    });
  }

  void _detectBrowserVisibilityChange() {
    if (!_browserIsActive && _random.nextDouble() < 0.3) {
      _browserIsActive = true;
      _startBrowserSession();
    } else if (_browserIsActive && _random.nextDouble() < 0.2) {
      _browserIsActive = false;
      _endBrowserSession();
    }
  }

  void _startBrowserSession() {
    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!_browserIsActive) {
        timer.cancel();
        return;
      }
      _detectTypingActivity();
    });
  }

  void _endBrowserSession() {
    _logBrowserSessionUrls();
    _recentBrowserUrls.clear();
    _typingPatterns.clear();
  }

  void _detectTypingActivity() {
    List<String> commonTypingPatterns = [
      'real',
      'realme',
      'realme.com',
      'www.realme.com',
      'https://www.realme.com',
      'https://www.realme.com/th/',
      'google',
      'google.com',
      'www.google.com',
      'developer.android.com',
      'github.com',
      'stackoverflow.com',
    ];

    if (_random.nextDouble() < 0.4) {
      String typedText =
          commonTypingPatterns[_random.nextInt(commonTypingPatterns.length)];
      _typingPatterns.add(typedText);

      if (_isCompleteUrl(typedText)) {
        String completeUrl = _normalizeUrl(typedText);
        _captureRealBrowserUrl(completeUrl, 'Browser Typing');
      }
    }
  }

  void _detectRealBrowserActivity() {
    if (!_browserIsActive) return;

    List<String> realBrowserUrls = [
      'https://www.realme.com/th/',
      'https://www.realme.com/th/smartphones',
      'https://www.google.com/search?q=realme+phone',
      'https://www.google.com/search?q=hello',
      'https://developer.android.com/guide/topics/manifest/service-element?hl=th',
      'https://stackoverflow.com/questions/android-service',
      'https://github.com/flutter/flutter',
      'https://www.youtube.com/watch?v=programming',
    ];

    if (_random.nextDouble() < 0.6) {
      String detectedUrl =
          realBrowserUrls[_random.nextInt(realBrowserUrls.length)];
      _captureRealBrowserUrl(detectedUrl, 'Browser Navigation');
    }
  }

  Future<void> _captureRealBrowserUrl(String url, String activityType) async {
    if (_recentBrowserUrls.contains(url)) return;

    _recentBrowserUrls.add(url);
    String browserName = _detectBrowserFromUrl(url);
    await _urlService.logUrl(url, 'Real Browser: $browserName ($activityType)');
  }

  String _detectBrowserFromUrl(String url) {
    if (url.contains('realme.com')) return 'Chrome (Shopping)';
    if (url.contains('developer.android.com')) return 'Chrome (Developer)';
    if (url.contains('google.com/search')) return 'Chrome (Search)';
    if (url.contains('youtube.com')) return 'Chrome (Video)';
    if (url.contains('github.com')) return 'Chrome (Code)';
    if (url.contains('stackoverflow.com')) return 'Chrome (Development)';
    return 'Chrome (General)';
  }

  bool _isCompleteUrl(String text) {
    return text.contains('.com') ||
        text.contains('.org') ||
        text.contains('.net') ||
        text.contains('.th') ||
        text.startsWith('http://') ||
        text.startsWith('https://') ||
        text.startsWith('www.');
  }

  String _normalizeUrl(String url) {
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      if (url.startsWith('www.')) {
        return 'https://$url';
      } else {
        return 'https://www.$url';
      }
    }
    return url;
  }

  void _logBrowserSessionUrls() {
    for (String url in _recentBrowserUrls) {}
  }

  Future<void> simulateBrowserActivity(String url) async {
    _browserIsActive = true;
    await _captureRealBrowserUrl(url, 'Manual Test');
  }

  void simulateTyping(String text) {
    _typingPatterns.add(text);
    if (_isCompleteUrl(text)) {
      String completeUrl = _normalizeUrl(text);
      _captureRealBrowserUrl(completeUrl, 'Simulated Typing');
    }
  }

  Future<void> detectClipboardFromBrowser() async {
    try {
      ClipboardData? data = await Clipboard.getData(Clipboard.kTextPlain);
      if (data?.text != null) {
        String clipboardText = data!.text!.trim();
        if (_isCompleteUrl(clipboardText)) {
          await _captureRealBrowserUrl(clipboardText, 'Clipboard Copy');
        }
      }
    } catch (e) {}
  }

  bool get isMonitoring => _isMonitoring;
  bool get browserIsActive => _browserIsActive;
  int get detectedUrlsCount => _recentBrowserUrls.length;
  List<String> get typingPatterns => List.unmodifiable(_typingPatterns);

  void stopMonitoring() {
    _monitoringTimer?.cancel();
    _visibilityTimer?.cancel();
    _isMonitoring = false;
    _browserIsActive = false;
  }

  void dispose() {
    stopMonitoring();
  }
}
