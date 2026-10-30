const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("toocomfy", {
  appInfo: () => ipcRenderer.invoke("app:info"),
});
