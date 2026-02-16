(() => {
  const getMermaidTheme = () =>
    document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default";

  const getThemeVariables = () =>
    getMermaidTheme() === "dark"
      ? {
          background: "transparent",
          primaryColor: "#1f2a44",
          primaryTextColor: "#ffffff",
          primaryBorderColor: "#8c9eff",
          lineColor: "#c5cae9",
          secondaryColor: "#151b2e",
          tertiaryColor: "#151b2e",
          fontFamily:
            "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        }
      : {
          background: "transparent",
          primaryColor: "#e8eaf6",
          primaryTextColor: "#1a237e",
          primaryBorderColor: "#3f51b5",
          lineColor: "#3f51b5",
          secondaryColor: "#f5f5f5",
          tertiaryColor: "#ffffff",
          fontFamily:
            "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        };

  const snapshotSource = (el) => {
    if (!el.dataset.mermaidSource) el.dataset.mermaidSource = el.textContent;
  };

  const restoreSource = (el) => {
    el.textContent = el.dataset.mermaidSource ?? el.textContent ?? "";
  };

  const render = async () => {
    if (!window.mermaid) return;

    const nodes = Array.from(document.querySelectorAll(".mermaid"));
    if (nodes.length === 0) return;

    nodes.forEach(snapshotSource);
    nodes.forEach(restoreSource);

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: getThemeVariables(),
      flowchart: { htmlLabels: true },
    });

    try {
      await window.mermaid.run({ nodes });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Mermaid render failed:", err);
    }
  };

  const scheduleRender = () => window.requestAnimationFrame(() => void render());

  if (window.document$?.subscribe) {
    window.document$.subscribe(() => scheduleRender());
  } else {
    document.addEventListener("DOMContentLoaded", () => scheduleRender());
  }

  const observer = new MutationObserver(() => scheduleRender());
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
})();
