
(function () {
  "use strict";
  var PLACEHOLDER = "TODO-LINK";
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.querySelectorAll('a[href*="' + PLACEHOLDER + '"]').forEach(function (a) {
    var local = a.getAttribute("data-local");
    if (local) {
      a.setAttribute("href", local);
      a.removeAttribute("data-local");
      return;
    }
    a.setAttribute("aria-disabled", "true");
    a.setAttribute("tabindex", "-1");
    a.addEventListener("click", function (e) { e.preventDefault(); });
  });

  var loop = document.querySelector(".hero__loop");
  if (loop && !reducedMotion && window.innerWidth >= 768 &&
      !(navigator.connection && navigator.connection.saveData)) {
    ["webm", "mp4"].forEach(function (ext) {
      var src = loop.getAttribute("data-" + ext);
      if (!src) return;
      var s = document.createElement("source");
      s.src = src;
      s.type = "video/" + ext;
      loop.appendChild(s);
    });
    loop.addEventListener("loadeddata", function () { loop.classList.add("is-ready"); });
    loop.load();
    loop.play().catch(function () {});
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) loop.play().catch(function () {}); else loop.pause();
        });
      }, { threshold: 0.1 }).observe(loop);
    }
  }

  var frame = document.querySelector("#promo");
  if (frame) {
    var play = frame.querySelector(".play");
    play.addEventListener("click", function () {
      var yt = frame.getAttribute("data-youtube") || "";
      var fallback = frame.getAttribute("data-fallback") || "";
      var el;
      if (yt && yt.indexOf(PLACEHOLDER) === -1) {
        el = document.createElement("iframe");
        el.src = "https://www.youtube-nocookie.com/embed/" + yt + "?autoplay=1&rel=0&modestbranding=1";
        el.allow = "accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture";
        el.allowFullscreen = true;
        el.title = "MambaSplat-4D overview video";
      } else if (fallback) {
        el = document.createElement("video");
        el.src = fallback;
        el.controls = true;
        el.autoplay = true;
        el.playsInline = true;
        el.addEventListener("error", function () {
          var n = document.createElement("p");
          n.className = "notice";
          n.textContent = "The video has not been uploaded yet.";
          frame.appendChild(n);
        });
      }
      if (el) {
        play.remove();
        frame.appendChild(el);
      }
    });
  }

  if (reducedMotion) {
    document.querySelectorAll("video[autoplay]").forEach(function (v) {
      v.removeAttribute("autoplay");
      v.controls = true;
      v.pause();
    });
  }

  var classButtons = Array.prototype.slice.call(document.querySelectorAll("[data-class-select] [data-class]"));
  function selectClass(name) {
    classButtons.forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-class") === name ? "true" : "false");
    });
    document.querySelectorAll("[data-class-media]").forEach(function (el) {
      var base = el.getAttribute("data-class-media").replace("{class}", name);
      if (el.tagName === "IMG") { el.src = base + ".webp"; return; }
      var sources = el.querySelectorAll("source");
      sources[0].src = base + ".webm";
      sources[1].src = base + ".mp4";
      el.poster = base + ".jpg";
      el.load();
      if (!reducedMotion) el.play().catch(function () {});
    });
  }
  classButtons.forEach(function (b) {
    b.addEventListener("click", function () { selectClass(b.getAttribute("data-class")); });
  });

  document.querySelectorAll("button[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var el = document.querySelector(btn.getAttribute("data-copy"));
      if (!el) return;
      var text = el.textContent;
      function done() {
        var old = btn.textContent;
        btn.textContent = "Copied";
        btn.classList.add("is-done");
        setTimeout(function () { btn.textContent = old; btn.classList.remove("is-done"); }, 1500);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { legacy(); });
      } else {
        legacy();
      }
      function legacy() {
        var r = document.createRange();
        r.selectNodeContents(el);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
        try { document.execCommand("copy"); done(); } catch (e) {}
        sel.removeAllRanges();
      }
    });
  });

  var bar = document.querySelector(".topbar");
  if (bar) {
    var onScroll = function () { bar.classList.toggle("topbar--scrolled", window.scrollY > 4); };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
})();
