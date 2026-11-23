// Shared settings for every page. Loaded before the page's own script.
// On localhost the pages talk to the local server, anywhere else to the Render backend.
(function () {
  const LOCAL = "http://127.0.0.1:8000";
  const DEPLOYED = "https://climate-resillient-tree-plantation.onrender.com";
  const host = location.hostname;
  window.TREE_API = (host === "localhost" || host === "127.0.0.1") ? LOCAL : DEPLOYED;

  // A free Render service sleeps after 15 minutes without traffic and needs about a minute to wake up.
  // The login page calls this so the server is already awake when the person starts asking questions.
  // mode "no-cors": the answer is not read, the request only has to reach the server.
  window.wakeApi = function () {
    try { fetch(window.TREE_API + "/health", { mode: "no-cors" }).catch(function () {}); } catch (e) { /* ignore */ }
  };
})();