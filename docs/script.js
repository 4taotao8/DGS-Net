const menuButton = document.querySelector('.menu-toggle');
const menu = document.querySelector('.primary-nav');

if (menuButton && menu) {
  menuButton.addEventListener('click', () => {
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!isOpen));
    menuButton.setAttribute('aria-label', isOpen ? 'Open navigation' : 'Close navigation');
    menu.classList.toggle('is-open', !isOpen);
  });

  menu.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      menuButton.setAttribute('aria-expanded', 'false');
      menuButton.setAttribute('aria-label', 'Open navigation');
      menu.classList.remove('is-open');
    });
  });
}

const heroVideo = document.querySelector('.hero-video');
const heroPlayButton = document.querySelector('.hero-play-button');
const heroChapters = [...document.querySelectorAll('.hero-chapter')];
if (heroVideo && heroPlayButton) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let manuallyPaused = false;
  let visibilityPausePending = false;
  let pendingChapterTime = null;

  const updateActiveChapter = () => {
    let activeIndex = 0;
    heroChapters.forEach((chapter, index) => {
      if (heroVideo.currentTime >= Number(chapter.dataset.time)) activeIndex = index;
    });
    heroChapters.forEach((chapter, index) => {
      if (index === activeIndex) chapter.setAttribute('aria-current', 'true');
      else chapter.removeAttribute('aria-current');
    });
  };

  const playFromChapter = () => {
    if (pendingChapterTime === null || heroVideo.readyState < 1) return;
    heroVideo.currentTime = pendingChapterTime;
    pendingChapterTime = null;
    updateActiveChapter();
    heroVideo.play().catch(() => { heroPlayButton.hidden = false; });
  };

  heroChapters.forEach((chapter) => {
    chapter.addEventListener('click', () => {
      pendingChapterTime = Number(chapter.dataset.time);
      if (heroVideo.readyState < 1) heroVideo.load();
      else playFromChapter();
    });
  });
  heroVideo.addEventListener('loadedmetadata', playFromChapter);
  heroVideo.addEventListener('timeupdate', updateActiveChapter);
  heroVideo.addEventListener('seeking', updateActiveChapter);

  heroPlayButton.addEventListener('click', () => {
    heroVideo.play().catch(() => { heroPlayButton.hidden = false; });
  });

  heroVideo.addEventListener('play', () => {
    manuallyPaused = false;
    heroPlayButton.hidden = true;
  });

  heroVideo.addEventListener('pause', () => {
    if (visibilityPausePending) {
      visibilityPausePending = false;
    } else {
      manuallyPaused = true;
    }
    heroPlayButton.hidden = false;
  });

  if (!reducedMotion.matches && 'IntersectionObserver' in window) {
    const videoObserver = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !manuallyPaused) {
        heroVideo.play().catch(() => {});
      } else if (!entry.isIntersecting && !heroVideo.paused) {
        visibilityPausePending = true;
        heroVideo.pause();
      }
    }, { threshold: 0.15 });
    videoObserver.observe(heroVideo);
  }
}
