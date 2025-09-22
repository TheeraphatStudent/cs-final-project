import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_accessibility_service/flutter_accessibility_service.dart';

class AccessibilityService {
  Future<void> requestPermission() async {
    await FlutterAccessibilityService.requestAccessibilityPermission();
  }

  Stream<AccessibilityEvent> get events =>
      FlutterAccessibilityService.accessStream;
}
