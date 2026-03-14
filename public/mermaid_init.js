// public/mermaid_init.js
// Loaded globally by Chainlit via custom_js in .chainlit/config.toml.
// Initialises mermaid.js from CDN and watches for new .mermaid elements
// added dynamically by React (graph updates arrive after initial page load).

(function () {
  function loadAndInit() {
    if (window.__mermaidReady) return;
    window.__mermaidReady = true;

    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
    s.onload = function () {
      mermaid.initialize({
        startOnLoad: false,
        theme: "dark",
        securityLevel: "loose",
      });
      renderAll();
      // Watch for elements added by React re-renders / message updates.
      var observer = new MutationObserver(function (mutations) {
        var needsRender = false;
        mutations.forEach(function (m) {
          m.addedNodes.forEach(function (n) {
            if (n.nodeType === 1) {
              if (
                n.classList.contains("mermaid") ||
                n.querySelector(".mermaid")
              ) {
                needsRender = true;
              }
            }
          });
        });
        if (needsRender) renderAll();
      });
      observer.observe(document.body, { childList: true, subtree: true });
    };
    document.head.appendChild(s);
  }

  function renderAll() {
    if (!window.mermaid) return;
    // Reset processed flag so mermaid re-renders updated content.
    document.querySelectorAll(".mermaid[data-processed]").forEach(function (el) {
      el.removeAttribute("data-processed");
    });
    mermaid.run({ querySelector: ".mermaid" });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadAndInit);
  } else {
    loadAndInit();
  }
})();
