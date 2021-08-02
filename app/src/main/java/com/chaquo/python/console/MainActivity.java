package com.chaquo.python.console;

import android.app.*;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.utils.*;

public class MainActivity extends PythonConsoleActivity {

    @Override protected Class<? extends Task> getTaskClass() {
        return Task.class;
    }

    public static class Task extends PythonConsoleActivity.Task {
        public Task(Application app) {
            super(app);
        }

        @Override public void run() {

            py.getModule("train").callAttr("train_function");
        }
    }
}
