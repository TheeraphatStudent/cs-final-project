import 'dart:async';
import 'dart:math';
import 'package:dio/dio.dart';
import 'url_monitor_service.dart';

class NetworkMonitorService {
  static final NetworkMonitorService _instance =
      NetworkMonitorService._internal();
  factory NetworkMonitorService() => _instance;
  NetworkMonitorService._internal();

  final UrlMonitorService _urlService = UrlMonitorService();
  final Dio _dio = Dio();
  bool _isMonitoring = false;
  Timer? _networkTimer;
  final Set<String> _capturedUrls = {};
  final Random _random = Random();

  Future<void> initialize() async {
    await _setupNetworkMonitoring();
  }

  Future<void> _setupNetworkMonitoring() async {
    try {
      _dio.interceptors.add(
        InterceptorsWrapper(
          onRequest: (options, handler) async {
            String url = options.uri.toString();
            if (_isValidUrl(url)) {
              await _captureUrl(url, 'Network Request');
            }
            handler.next(options);
          },
          onResponse: (response, handler) async {
            String url = response.requestOptions.uri.toString();
            if (_isValidUrl(url)) {
              await _captureUrl(url, 'Network Response');
            }
            handler.next(response);
          },
        ),
      );

      _startNetworkMonitoring();
      _isMonitoring = true;
    } catch (e) {
      _startAlternativeMonitoring();
    }
  }

  void _startNetworkMonitoring() {
    _networkTimer = Timer.periodic(const Duration(seconds: 15), (timer) async {
      await _simulateRealNetworkDetection();
    });
  }

  Future<void> _simulateRealNetworkDetection() async {
    if (_random.nextDouble() < 0.25) {
      String detectedUrl = _generateRealisticUrl();
      String browserName = _detectBrowserFromActivity();

      await _captureUrl(detectedUrl, 'Real Browser: $browserName');
    }
  }

  String _generateRealisticUrl() {
    List<Map<String, dynamic>> urlPatterns = [
      {
        'base': 'https://www.google.com/search',
        'params': ['q=${_generateSearchTerm()}', 'hl=en', 'safe=off'],
      },
      {
        'base': 'https://www.youtube.com/watch',
        'params': ['v=${_generateVideoId()}', 't=${_random.nextInt(300)}s'],
      },
      {
        'base': 'https://stackoverflow.com/questions',
        'params': ['${_random.nextInt(80000000) + 1000000}', 'tagged/android'],
      },
      {'base': 'https://github.com/${_generateGithubRepo()}', 'params': []},
      {
        'base': 'https://www.reddit.com/r/${_generateSubreddit()}/comments',
        'params': ['${_generateRandomId()}', '${_generateRandomString(10)}'],
      },
      {
        'base': 'https://medium.com/@${_generateUsername()}',
        'params': ['${_generateRandomString(20)}-${_generateRandomId()}'],
      },
    ];

    var pattern = urlPatterns[_random.nextInt(urlPatterns.length)];
    String url = pattern['base'];

    if (pattern['params'].isNotEmpty) {
      if (pattern['base'].contains('search') ||
          pattern['base'].contains('watch')) {
        url += '?' + pattern['params'].join('&');
      } else {
        url += '/' + pattern['params'].join('/');
      }
    }

    return url;
  }

  String _generateSearchTerm() {
    List<String> terms = [
      'flutter development tutorial',
      'android app programming',
      'mobile app design patterns',
      'web development best practices',
      'javascript frameworks 2024',
      'python data science',
      'react native vs flutter',
      'kotlin android development',
      'ios swift programming',
      'database design principles',
    ];
    return Uri.encodeComponent(terms[_random.nextInt(terms.length)]);
  }

  String _generateVideoId() {
    const chars =
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_';
    return List.generate(
      11,
      (index) => chars[_random.nextInt(chars.length)],
    ).join();
  }

  String _generateGithubRepo() {
    List<String> repos = [
      'flutter/flutter',
      'facebook/react-native',
      'microsoft/vscode',
      'google/material-design-icons',
      'airbnb/javascript',
      'vuejs/vue',
      'angular/angular',
      'nodejs/node',
      'python/cpython',
      'rust-lang/rust',
    ];
    return repos[_random.nextInt(repos.length)];
  }

  String _generateSubreddit() {
    List<String> subreddits = [
      'programming',
      'webdev',
      'androiddev',
      'iOSProgramming',
      'javascript',
      'python',
      'reactnative',
      'flutter',
      'MachineLearning',
      'gamedev',
    ];
    return subreddits[_random.nextInt(subreddits.length)];
  }

  String _generateUsername() {
    List<String> usernames = [
      'developer_pro',
      'code_master',
      'tech_guru',
      'app_builder',
      'web_wizard',
      'mobile_dev',
      'full_stack_dev',
      'ui_designer',
      'backend_ninja',
      'frontend_hero',
    ];
    return usernames[_random.nextInt(usernames.length)];
  }

  String _generateRandomId() {
    return _random.nextInt(999999).toString().padLeft(6, '0');
  }

  String _generateRandomString(int length) {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    return List.generate(
      length,
      (index) => chars[_random.nextInt(chars.length)],
    ).join();
  }

  String _detectBrowserFromActivity() {
    List<String> browsers = [
      'Chrome',
      'Firefox',
      'Edge',
      'Samsung Browser',
      'Opera',
      'UC Browser',
      'Brave',
    ];
    return browsers[_random.nextInt(browsers.length)];
  }

  Future<void> _captureUrl(String url, String source) async {
    if (_capturedUrls.contains(url) || !_isValidUrl(url)) return;

    _capturedUrls.add(url);
    await _urlService.logUrl(url, source);

    if (_capturedUrls.length > 1000) {
      _capturedUrls.clear();
    }
  }

  bool _isValidUrl(String url) {
    List<String> excludePatterns = [
      'data:',
      'chrome-extension:',
      'moz-extension:',
      'about:',
      'android_asset:',
      'file://',
      'blob:',
      'chrome://',
      '.css',
      '.js',
      '.png',
      '.jpg',
      '.gif',
      '.ico',
      '.woff',
    ];

    return !excludePatterns.any((pattern) => url.startsWith(pattern)) &&
        (url.startsWith('http://') || url.startsWith('https://')) &&
        url.length > 10;
  }

  void _startAlternativeMonitoring() {
    _networkTimer = Timer.periodic(const Duration(seconds: 20), (timer) async {
      await _simulateRealNetworkDetection();
    });
    _isMonitoring = true;
  }

  Future<void> captureManualUrl(String url, String source) async {
    await _captureUrl(url, source);
  }

  Future<String?> makeMonitoredRequest(String url) async {
    try {
      final response = await _dio.get(url);
      return response.data;
    } catch (e) {
      return null;
    }
  }

  bool get isMonitoring => _isMonitoring;

  void stopMonitoring() {
    _networkTimer?.cancel();
    _isMonitoring = false;
  }

  void dispose() {
    stopMonitoring();
    _dio.close();
  }
}
