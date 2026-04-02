// ── Trace line cycling animation ──────────────────────────────────────
function cycleTrace() {
    const lines = document.querySelectorAll('.trace-line');
    if (!lines.length) return;

    let current = 0;
    lines.forEach(l => { l.classList.remove('trace-line--active', 'trace-line--done'); });
    lines[0].classList.add('trace-line--active');

    const interval = setInterval(() => {
        lines[current].classList.remove('trace-line--active');
        lines[current].classList.add('trace-line--done');
        current++;
        if (current >= lines.length) {
            clearInterval(interval);
            setTimeout(() => {
                lines.forEach(l => { l.classList.remove('trace-line--active', 'trace-line--done'); });
                cycleTrace();
            }, 2500);
            return;
        }
        lines[current].classList.add('trace-line--active');
    }, 900);
}

// ── Hamburger menu ────────────────────────────────────────────────────
const hamburger = document.querySelector('.nav-hamburger');
const navLinks  = document.querySelector('.nav-links');
if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
        navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
    });
}

// ── Scroll reveal ─────────────────────────────────────────────────────
function revealOnScroll() {
    const els = document.querySelectorAll('.feat-card, .step, .dash-card');
    const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.style.opacity = '1';
                e.target.style.transform = 'translateY(0)';
                io.unobserve(e.target);
            }
        });
    }, { threshold: 0.1 });

    els.forEach((el, i) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = `opacity 0.5s ease ${i * 0.07}s, transform 0.5s ease ${i * 0.07}s`;
        io.observe(el);
    });
}

// ── Auto-dismiss flash ─────────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(f => {
    setTimeout(() => { f.style.opacity = '0'; f.style.transition = 'opacity 0.4s'; setTimeout(() => f.remove(), 400); }, 4000);
});

// ── Init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    cycleTrace();
    revealOnScroll();
});
