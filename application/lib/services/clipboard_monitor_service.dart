import 'dart:async';
import 'package:flutter/services.dart';
import 'url_monitor_service.dart';

class ClipboardMonitorService {
  static final ClipboardMonitorService _instance =
      ClipboardMonitorService._internal();
  factory ClipboardMonitorService() => _instance;
  ClipboardMonitorService._internal();

  final UrlMonitorService _urlService = UrlMonitorService();
  Timer? _clipboardTimer;
  String _lastClipboardContent = '';
  bool _isMonitoring = false;
  final Set<String> _processedUrls = {};

  void startMonitoring() {
    if (_isMonitoring) return;

    _isMonitoring = true;
    _clipboardTimer = Timer.periodic(const Duration(milliseconds: 500), (
      timer,
    ) {
      _checkClipboard();
    });
  }

  Future<void> _checkClipboard() async {
    try {
      ClipboardData? data = await Clipboard.getData(Clipboard.kTextPlain);
      if (data?.text != null) {
        String currentContent = data!.text!.trim();

        if (currentContent != _lastClipboardContent &&
            _isUrl(currentContent) &&
            !_processedUrls.contains(currentContent)) {
          _lastClipboardContent = currentContent;
          _processedUrls.add(currentContent);

          String browserSource = _detectBrowserSource(currentContent);
          await _urlService.logUrl(currentContent, 'Clipboard: $browserSource');

          if (_processedUrls.length > 500) {
            _processedUrls.clear();
          }
        }
      }
    } catch (e) {}
  }

  String _detectBrowserSource(String url) {
    if (url.contains('realme.com') ||
        url.contains('shopping') ||
        url.contains('product')) {
      return 'Chrome (Shopping)';
    } else if (url.contains('developer.android.com')) {
      return 'Chrome (Developer)';
    } else if (url.contains('youtube.com') || url.contains('watch')) {
      return 'Chrome (Video)';
    } else if (url.contains('google.com/search')) {
      return 'Chrome (Search)';
    } else if (url.contains('stackoverflow.com')) {
      return 'Chrome (Development)';
    } else if (url.contains('github.com')) {
      return 'Chrome (Code)';
    } else if (url.contains('reddit.com')) {
      return 'Chrome (Social)';
    } else if (url.contains('facebook.com') ||
        url.contains('instagram.com') ||
        url.contains('twitter.com')) {
      return 'Chrome (Social Media)';
    } else {
      return 'Browser Copy/Paste';
    }
  }

  bool _isUrl(String text) {
    if (text.length < 7) return false;

    RegExp urlRegex = RegExp(
      r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
      caseSensitive: false,
    );

    RegExp simpleUrlRegex = RegExp(
      r'^(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
      caseSensitive: false,
    );

    bool hasUrlPattern =
        urlRegex.hasMatch(text) ||
        simpleUrlRegex.hasMatch(text) ||
        text.startsWith('http://') ||
        text.startsWith('https://') ||
        text.contains('www.') ||
        text.contains('.com') ||
        text.contains('.org') ||
        text.contains('.net') ||
        text.contains('.th') ||
        text.contains('.dev') ||
        text.contains('.io');

    bool isMobileUrl =
        text.contains('m.') ||
        text.contains('mobile.') ||
        text.contains('amp.') ||
        text.contains('app.');

    return hasUrlPattern || isMobileUrl;
  }

  Future<void> checkClipboardNow() async {
    await _checkClipboard();
  }

  Future<void> simulateUrlDetection(String url) async {
    String browserSource = _detectBrowserSource(url);
    await _urlService.logUrl(url, 'Simulated: $browserSource');
  }

  bool get isMonitoring => _isMonitoring;

  void stopMonitoring() {
    _clipboardTimer?.cancel();
    _isMonitoring = false;
  }

  void dispose() {
    stopMonitoring();
  }
}
