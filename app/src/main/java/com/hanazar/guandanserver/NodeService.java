package com.hanazar.guandanserver;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.IBinder;
import android.util.Log;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * 前台服务：复制家庭版网页资源，并启动内置 Python 家庭版服务器。
 */
public class NodeService extends Service {
    private static final String TAG = "GuandanServer";
    private static final String CHANNEL_ID = "guandan_server";
    private static final int NOTIF_ID = 1;
    public static final int PORT = 5000;
    private static boolean serverStarted = false;
    private static volatile String startupError = null;

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(NOTIF_ID, buildNotification("掼蛋服务器启动中…"));
        if (!serverStarted) {
            serverStarted = true;
            startupError = null;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        String dataDir = getApplicationContext().getFilesDir().getAbsolutePath();
                        String webDir = dataDir + "/family-web";
                        if (wasAPKUpdated()) {
                            File webDirReference = new File(webDir);
                            if (webDirReference.exists()) {
                                deleteFolderRecursively(webDirReference);
                            }
                            copyAssetFolder(getApplicationContext().getAssets(), "family-web", webDir);
                            saveLastUpdateTime();
                        }
                        if (!Python.isStarted()) {
                            Python.start(new AndroidPlatform(getApplicationContext()));
                        }
                        Log.i(TAG, "starting family Python server on port " + PORT);
                        Python.getInstance().getModule("family_backend.mobile_server")
                                .callAttr("start", dataDir, PORT);
                    } catch (Throwable t) {
                        Log.e(TAG, "family server start failed", t);
                        startupError = t.getClass().getSimpleName() + ": "
                                + (t.getMessage() == null ? "未知启动错误" : t.getMessage());
                        updateNotification(getApplicationContext(), "服务器启动失败");
                        serverStarted = false;
                        stopSelf();
                    }
                }
            }).start();
        }
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    /** 供首页显示内嵌 Python 的启动失败原因，避免只看到网页拒绝连接。 */
    public static String getStartupError() {
        return startupError;
    }

    /** 更新前台通知文案（MainActivity 拿到局域网地址后调用） */
    public static void updateNotification(Context context, String text) {
        NotificationManager nm = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) {
            Notification.Builder b = new Notification.Builder(context, CHANNEL_ID)
                    .setContentTitle("掼蛋服务器")
                    .setContentText(text)
                    .setSmallIcon(R.drawable.ic_stat_card)
                    .setOngoing(true);
            nm.notify(NOTIF_ID, b.build());
        }
    }

    private Notification buildNotification(String text) {
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("掼蛋服务器")
                .setContentText(text)
                .setSmallIcon(R.drawable.ic_stat_card)
                .setOngoing(true)
                .build();
    }

    private void createChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "掼蛋服务器", NotificationManager.IMPORTANCE_LOW);
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.createNotificationChannel(channel);
    }

    // ---- assets 复制辅助 ----

    private static boolean deleteFolderRecursively(File file) {
        try {
            boolean res = true;
            File[] children = file.listFiles();
            if (children != null) {
                for (File childFile : children) {
                    if (childFile.isDirectory()) {
                        res &= deleteFolderRecursively(childFile);
                    } else {
                        res &= childFile.delete();
                    }
                }
            }
            res &= file.delete();
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean copyAssetFolder(AssetManager assetManager, String fromAssetPath, String toPath) {
        try {
            String[] files = assetManager.list(fromAssetPath);
            boolean res = true;
            if (files == null || files.length == 0) {
                res &= copyAsset(assetManager, fromAssetPath, toPath);
            } else {
                new File(toPath).mkdirs();
                for (String file : files) {
                    res &= copyAssetFolder(assetManager, fromAssetPath + "/" + file, toPath + "/" + file);
                }
            }
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean copyAsset(AssetManager assetManager, String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(fromAssetPath);
            File outFile = new File(toPath);
            if (!outFile.getParentFile().exists()) outFile.getParentFile().mkdirs();
            out = new FileOutputStream(outFile);
            copyFile(in, out);
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    // ---- APK 更新检测：只在升级后重新复制 assets ----

    private boolean wasAPKUpdated() {
        SharedPreferences prefs = getApplicationContext().getSharedPreferences("NODEJS_MOBILE_PREFS", Context.MODE_PRIVATE);
        long previousLastUpdateTime = prefs.getLong("NODEJS_MOBILE_APK_LastUpdateTime", 0);
        long lastUpdateTime = 1;
        try {
            PackageInfo packageInfo = getApplicationContext().getPackageManager()
                    .getPackageInfo(getApplicationContext().getPackageName(), 0);
            lastUpdateTime = packageInfo.lastUpdateTime;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
        return (lastUpdateTime != previousLastUpdateTime);
    }

    private void saveLastUpdateTime() {
        long lastUpdateTime = 1;
        try {
            PackageInfo packageInfo = getApplicationContext().getPackageManager()
                    .getPackageInfo(getApplicationContext().getPackageName(), 0);
            lastUpdateTime = packageInfo.lastUpdateTime;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
        SharedPreferences prefs = getApplicationContext().getSharedPreferences("NODEJS_MOBILE_PREFS", Context.MODE_PRIVATE);
        prefs.edit().putLong("NODEJS_MOBILE_APK_LastUpdateTime", lastUpdateTime).apply();
    }
}
