import 'dart:async';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class UrlLogEntry {
  final String url;
  final String source;
  final DateTime timestamp;

  UrlLogEntry({
    required this.url,
    required this.source,
    required this.timestamp,
  });

  factory UrlLogEntry.fromJson(Map<String, dynamic> json) {
    return UrlLogEntry(
      url: json['url'],
      source: json['source'],
      timestamp: DateTime.parse(json['timestamp']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'url': url,
      'source': source,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  String get formattedTime {
    final now = DateTime.now();
    final difference = now.difference(timestamp);

    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }
}

class UrlMonitorService {
  static final UrlMonitorService _instance = UrlMonitorService._internal();
  factory UrlMonitorService() => _instance;
  UrlMonitorService._internal();

  final StreamController<UrlLogEntry> _urlStreamController =
      StreamController<UrlLogEntry>.broadcast();
  final List<UrlLogEntry> _urlHistory = [];
  Timer? _backgroundTimer;

  Stream<UrlLogEntry> get urlStream => _urlStreamController.stream;
  List<UrlLogEntry> get urlHistory => List.unmodifiable(_urlHistory);

  Future<void> initialize() async {
    await _loadUrlHistory();
    _startBackgroundService();
  }

  void _startBackgroundService() {
    _backgroundTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      _simulateBackgroundActivity();
    });
  }

  void _simulateBackgroundActivity() {
    List<String> backgroundUrls = [
      'https://www.google.com/search?q=background+activity',
      'https://www.youtube.com/watch?v=background',
      'https://stackoverflow.com/questions/background',
      'https://github.com/flutter/background',
    ];

    if (DateTime.now().millisecond % 5 == 0) {
      String url =
          backgroundUrls[DateTime.now().second % backgroundUrls.length];
      logUrl(url, 'Background Activity');
    }
  }

  Future<void> logUrl(String url, String source) async {
    print('Logging URL: $url from $source');

    if (_urlHistory.any(
      (entry) => entry.url == url && entry.source == source,
    )) {
      return;
    }

    final entry = UrlLogEntry(
      url: url,
      source: source,
      timestamp: DateTime.now(),
    );

    _urlHistory.add(entry);
    _urlStreamController.add(entry);
    await _saveUrlHistory();
  }

  Future<void> _loadUrlHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final historyJson = prefs.getString('url_history');

      if (historyJson != null) {
        final List<dynamic> historyList = json.decode(historyJson);
        _urlHistory.clear();
        _urlHistory.addAll(
          historyList.map((item) => UrlLogEntry.fromJson(item)).toList(),
        );
      }
    } catch (e) {
      _urlHistory.clear();
    }
  }

  Future<void> _saveUrlHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final historyJson = json.encode(
        _urlHistory.map((entry) => entry.toJson()).toList(),
      );
      await prefs.setString('url_history', historyJson);
    } catch (e) {
      // Handle error silently
    }
  }

  List<UrlLogEntry> searchHistory(String query) {
    if (query.isEmpty) return _urlHistory;

    return _urlHistory.where((entry) {
      return entry.url.toLowerCase().contains(query.toLowerCase()) ||
          entry.source.toLowerCase().contains(query.toLowerCase());
    }).toList();
  }

  Future<void> clearHistory() async {
    _urlHistory.clear();
    await _saveUrlHistory();
  }

  void dispose() {
    _backgroundTimer?.cancel();
    _urlStreamController.close();
  }
}
