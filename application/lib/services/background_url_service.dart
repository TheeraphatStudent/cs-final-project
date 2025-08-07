import 'dart:async';
import 'package:flutter/services.dart';
import 'url_monitor_service.dart';

class BackgroundUrlService {
  static final BackgroundUrlService _instance =
      BackgroundUrlService._internal();
  factory BackgroundUrlService() => _instance;
  BackgroundUrlService._internal();

  static const String _channelName = 'com.example.application/url_clipboard';
  static const EventChannel _eventChannel = EventChannel(_channelName);
  static const MethodChannel _methodChannel = MethodChannel(_channelName);

  final UrlMonitorService _urlService = UrlMonitorService();
  StreamSubscription? _backgroundEventSubscription;
  bool _isListening = false;
  bool _serviceConnected = false;
  String _serviceStatus = 'disconnected';

  Future<void> initialize() async {
    await _startListening();
  }

  Future<void> _startListening() async {
    if (_isListening) return;

    try {
      _backgroundEventSubscription = _eventChannel
          .receiveBroadcastStream()
          .listen(
            _handleBackgroundEvent,
            onError: _handleBackgroundError,
            onDone: _handleBackgroundDone,
          );

      _isListening = true;
    } catch (e) {
      // Handle error silently
    }
  }

  Future<void> _handleBackgroundEvent(dynamic event) async {
    try {
      if (event is Map) {
        final eventType = event['type'] as String?;

        switch (eventType) {
          case 'url_detected':
            await _handleUrlDetected(event);
            break;
          case 'service_status':
            _handleServiceStatus(event);
            break;
          case 'browser_activity':
            _handleBrowserActivity(event);
            break;
          default:
            break;
        }
      }
    } catch (e) {
      // Handle error silently
    }
  }

  Future<void> _handleUrlDetected(Map event) async {
    final url = event['url'] as String?;
    final source = event['source'] as String?;
    final detectionMethod = event['detection_method'] as String?;

    if (url != null && source != null) {
      await _urlService.logUrl(url, source);
    }
  }

  void _handleServiceStatus(Map event) {
    final status = event['status'] as String?;
    final message = event['message'] as String?;

    if (status != null) {
      _serviceStatus = status;
      _serviceConnected = status == 'connected';
    }
  }

  void _handleBrowserActivity(Map event) {
    final browser = event['browser'] as String?;
    final packageName = event['package'] as String?;
  }

  void _handleBackgroundError(dynamic error) {
    _isListening = false;
    _serviceConnected = false;

    Timer(const Duration(seconds: 5), () {
      _startListening();
    });
  }

  void _handleBackgroundDone() {
    _isListening = false;
    _serviceConnected = false;
  }

  Future<bool> isAccessibilityServiceEnabled() async {
    try {
      final result = await _methodChannel.invokeMethod(
        'isAccessibilityServiceEnabled',
      );
      return result as bool? ?? false;
    } catch (e) {
      return false;
    }
  }

  Future<void> openAccessibilitySettings() async {
    try {
      await _methodChannel.invokeMethod('openAccessibilitySettings');
    } catch (e) {
      // Handle error silently
    }
  }

  Future<void> testBackgroundService() async {
    try {
      await _methodChannel.invokeMethod('testBackgroundService');
    } catch (e) {
      // Handle error silently
    }
  }

  bool get isListening => _isListening;
  bool get isServiceConnected => _serviceConnected;
  String get serviceStatus => _serviceStatus;

  void dispose() {
    _backgroundEventSubscription?.cancel();
    _isListening = false;
    _serviceConnected = false;
  }
}
