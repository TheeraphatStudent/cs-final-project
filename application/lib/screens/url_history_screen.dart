import 'package:flutter/material.dart';
import '../services/url_monitor_service.dart';

class UrlHistoryScreen extends StatefulWidget {
  const UrlHistoryScreen({super.key});

  @override
  State<UrlHistoryScreen> createState() => _UrlHistoryScreenState();
}

class _UrlHistoryScreenState extends State<UrlHistoryScreen> {
  final UrlMonitorService _urlService = UrlMonitorService();
  final TextEditingController _searchController = TextEditingController();
  List<UrlLogEntry> _filteredHistory = [];
  bool _isSearching = false;

  @override
  void initState() {
    super.initState();
    _loadHistory();
    _urlService.urlStream.listen((_) {
      if (mounted) {
        _loadHistory();
      }
    });
  }

  void _loadHistory() {
    setState(() {
      _filteredHistory = _searchController.text.isEmpty
          ? _urlService.urlHistory
          : _urlService.searchHistory(_searchController.text);
    });
  }

  void _onSearchChanged(String query) {
    setState(() {
      _filteredHistory = _urlService.searchHistory(query);
    });
  }

  Future<void> _clearHistory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear History'),
        content: const Text('Are you sure you want to clear all URL history?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await _urlService.clearHistory();
      _loadHistory();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('URL History'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: () {
              setState(() {
                _isSearching = !_isSearching;
                if (!_isSearching) {
                  _searchController.clear();
                  _loadHistory();
                }
              });
            },
            icon: Icon(_isSearching ? Icons.close : Icons.search),
          ),
          IconButton(
            onPressed: _clearHistory,
            icon: const Icon(Icons.delete_sweep),
          ),
        ],
        bottom: _isSearching
            ? PreferredSize(
                preferredSize: const Size.fromHeight(60),
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: TextField(
                    controller: _searchController,
                    onChanged: _onSearchChanged,
                    decoration: const InputDecoration(
                      hintText: 'Search URLs...',
                      prefixIcon: Icon(Icons.search),
                      border: OutlineInputBorder(),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    autofocus: true,
                  ),
                ),
              )
            : null,
      ),
      body: _filteredHistory.isEmpty
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.history, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'No URL history found',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Start browsing to see your URL history',
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            )
          : Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  color: Colors.blue[50],
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.blue),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Total URLs logged: ${_urlService.urlHistory.length}',
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                      ),
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
                                  : Colors.grey,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  snapshot.hasData
                                      ? Icons.circle
                                      : Icons.circle_outlined,
                                  size: 8,
                                  color: Colors.white,
                                ),
                                const SizedBox(width: 4),
                                Text(
                                  snapshot.hasData ? 'Active' : 'Monitoring',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: _filteredHistory.length,
                    itemBuilder: (context, index) {
                      final entry = _filteredHistory[index];
                      return Card(
                        margin: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        child: ListTile(
                          leading: _buildSourceIcon(entry.source),
                          title: Text(
                            entry.url,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(fontSize: 14),
                          ),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Source: ${entry.source}',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey[600],
                                ),
                              ),
                              Text(
                                entry.formattedTime,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey[600],
                                ),
                              ),
                            ],
                          ),
                          trailing: IconButton(
                            onPressed: () => _copyUrl(entry.url),
                            icon: const Icon(Icons.copy, size: 20),
                          ),
                          onTap: () => _showUrlDetails(entry),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
    );
  }

  Color _getSourceColor(String source) {
    switch (source.toLowerCase()) {
      case 'webview':
        return Colors.blue;
      case 'background':
        return Colors.green;
      case 'keyboard':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  IconData _getSourceIcon(String source) {
    switch (source.toLowerCase()) {
      case 'webview':
        return Icons.web;
      case 'background':
        return Icons.settings_backup_restore;
      case 'keyboard':
        return Icons.keyboard;
      default:
        return Icons.link;
    }
  }

  Widget _buildSourceIcon(String source) {
    IconData icon;
    Color color;

    if (source.startsWith('WebView')) {
      icon = Icons.web;
      color = Colors.blue;
    } else if (source.startsWith('Keyboard')) {
      icon = Icons.keyboard;
      color = Colors.orange;
    } else if (source.startsWith('Clipboard')) {
      if (source.contains('Chrome (Shopping)')) {
        icon = Icons.shopping_cart;
        color = Colors.green;
      } else if (source.contains('Chrome (Developer)')) {
        icon = Icons.code;
        color = Colors.purple;
      } else if (source.contains('Chrome (Video)')) {
        icon = Icons.play_circle;
        color = Colors.red;
      } else {
        icon = Icons.content_paste;
        color = Colors.green;
      }
    } else if (source.startsWith('Real Browser')) {
      if (source.contains('Typing')) {
        icon = Icons.keyboard_alt;
        color = Colors.indigo;
      } else if (source.contains('Navigation')) {
        icon = Icons.navigation;
        color = Colors.teal;
      } else {
        icon = Icons.web_asset;
        color = Colors.cyan;
      }
    } else if (source.startsWith('Network')) {
      icon = Icons.cloud;
      color = Colors.lightBlue;
    } else {
      icon = Icons.device_unknown;
      color = Colors.grey;
    }

    return Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Icon(icon, size: 16, color: color),
    );
  }

  void _copyUrl(String url) {
    // Implementation for copying URL to clipboard
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'URL copied: ${url.length > 50 ? '${url.substring(0, 50)}...' : url}',
        ),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  void _showUrlDetails(UrlLogEntry entry) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('URL Details'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('URL:', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              SelectableText(entry.url),
              const SizedBox(height: 16),
              const Text(
                'Source:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Text(entry.source),
              const SizedBox(height: 16),
              const Text(
                'Timestamp:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Text(entry.timestamp.toString()),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}
