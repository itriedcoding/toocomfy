const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

// The Python backend lives next to the packaged resources.
// In dev mode: project_root/backend + project_root/desktop/server_bootstrap.py
// In packaged mode: resources/backend + resources/server_bootstrap.py
const DEV_ROOT = path.join(__dirname, "..");
const PKG_ROOT = process.resourcesPath;
const ROOT = app.isPackaged ? PKG_ROOT : DEV_ROOT;
const BOOTSTRAP = app.isPackaged
  ? path.join(PKG_ROOT, "server_bootstrap.py")
  : path.join(DEV_ROOT, "desktop", "server_bootstrap.py");
let serverProc = null;
let mainWindow = null;

function freePort() {
  const net = require("net");
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const p = srv.address().port;
      srv.close(() => resolve(p));
    });
  });
}

async function startServer() {
  const port = await freePort();
  const env = { ...process.env, SMALLER_HOST: "127.0.0.1", SMALLER_PORT: String(port) };
  // Prefer a bundled Python if present, else system Python.
  const python = process.env.PYTHON || "python";
  serverProc = spawn(python, [BOOTSTRAP], {
    env,
    stdio: ["ignore", "pipe", "inherit"],
  });

  // Bootstrap prints the port on first line, then serves. Wait for the port.
  const portLine = await new Promise((resolve, reject) => {
    let buf = "";
    const t = setTimeout(reject, 60000);
    serverProc.stdout.on("data", (chunk) => {
      buf += chunk;
      const idx = buf.indexOf("\n");
      if (idx !== -1) {
        clearTimeout(t);
        resolve(parseInt(buf.slice(0, idx), 10));
      }
    });
    serverProc.on("error", reject);
  });

  // Confirm the server is actually listening before returning.
  await waitForHttp(`http://127.0.0.1:${port}/api/health`);
  return port;
}

function waitForHttp(url, tries = 60) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      http
        .get(url, (res) => {
          res.resume();
          res.on("end", resolve);
        })
        .on("error", () => {
          if (n <= 0) reject(new Error("server did not come up"));
          else setTimeout(() => attempt(n - 1), 500);
        });
    };
    attempt(tries);
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 720,
    minHeight: 480,
    backgroundColor: "#000000",
    title: "toocomfy",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  const port = await startServer();
  await mainWindow.loadURL(`http://127.0.0.1:${port}`);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (serverProc && !serverProc.killed) serverProc.kill();
});

ipcMain.handle("app:info", () => ({
  version: app.getVersion(),
  platform: process.platform,
  isPackaged: app.isPackaged,
}));
