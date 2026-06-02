// Attach a click handler to the theme's cookie-banner close button and
// remember the user's choice so the banner stays dismissed on subsequent
// visits. The pytorch_sphinx_theme2 template ships the markup and the
// close button but no JS to actually close it.
(function () {
  var STORAGE_KEY = "openenv-docs-cookie-banner-dismissed";

  function init() {
    var banner = document.querySelector(".cookie-banner-wrapper");
    if (!banner) return;

    try {
      if (window.localStorage && localStorage.getItem(STORAGE_KEY) === "1") {
        banner.style.display = "none";
        return;
      }
    } catch (_) {
      // localStorage may be unavailable (private mode, etc.) — fall through
      // to the click handler so the banner is at least closable for the session.
    }

    var closeBtn = banner.querySelector(".close-button");
    if (!closeBtn) return;

    closeBtn.addEventListener("click", function () {
      banner.style.display = "none";
      try {
        if (window.localStorage) localStorage.setItem(STORAGE_KEY, "1");
      } catch (_) {}
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
