(() => {
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const videos = document.querySelectorAll("video[data-autoplay]");

  if (!prefersReducedMotion && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting && entry.intersectionRatio >= 0.55) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }, { threshold: [0, 0.55] });

    videos.forEach((video) => observer.observe(video));
  }

  const copyButton = document.querySelector("[data-copy-citation]");
  const citation = document.querySelector("#bibtex code");
  const status = document.querySelector(".copy-status");

  if (copyButton && citation && status) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(citation.textContent.trim());
        status.textContent = "BibTeX copied.";
      } catch (_) {
        status.textContent = "Select the citation text to copy it.";
      }
    });
  }
})();
