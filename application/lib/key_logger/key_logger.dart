import 'dart:async';

import 'package:flutter/material.dart';

class KeyLogger extends StatefulWidget {
  const KeyLogger({super.key, required this.keyStream});

  final Stream<String> keyStream;

  @override
  State<KeyLogger> createState() => _KeyLoggerState();
}

class _KeyLoggerState extends State<KeyLogger> {
  String _text = '';
  StreamSubscription<String>? _keySubscription;

  @override
  void initState() {
    super.initState();
    _keySubscription = widget.keyStream.listen((key) {
      setState(() {
        if (key == 'Backspace') {
          if (_text.isNotEmpty) {
            _text = _text.substring(0, _text.length - 1);
          }
        } else {
          _text += key;
        }
      });
    });
  }

  @override
  void dispose() {
    _keySubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text(
          _text,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
      ),
    );
  }
}
