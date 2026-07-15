// SMART LENDER — Client Logic

document.addEventListener('DOMContentLoaded', function () {

    // 1. Theme Toggle
    const themeBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    function applyTheme(t) {
        document.documentElement.setAttribute('data-theme', t);
        localStorage.setItem('theme', t);
        if (themeIcon) themeIcon.className = t === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    applyTheme(localStorage.getItem('theme') || 'dark');

    if (themeBtn) {
        themeBtn.addEventListener('click', function () {
            const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            applyTheme(next);
            showToast(`Switched to ${next} mode`, 'info');
        });
    }

    // 2. Counter animation
    document.querySelectorAll('.counter-val').forEach(el => {
        const target = parseFloat(el.dataset.target);
        const isFloat = el.dataset.target.includes('.');
        let current = 0;
        const step = target / 80;
        const tick = () => {
            current = Math.min(current + step, target);
            el.textContent = isFloat ? current.toFixed(1) : Math.ceil(current);
            if (current < target) setTimeout(tick, 16);
        };
        tick();
    });

    // 3. Single predict form validation
    const predictForm = document.getElementById('predict-single-form');
    if (predictForm) {
        predictForm.addEventListener('submit', function (e) {
            let valid = true;

            const checks = [
                [document.getElementById('applicant_name'), v => v.trim() !== '', 'Name cannot be empty'],
                [document.getElementById('applicant_income'), v => !isNaN(v) && parseFloat(v) > 0, 'Income must be positive'],
                [document.getElementById('coapplicant_income'), v => !isNaN(v) && parseFloat(v) >= 0, 'Co-applicant income must be ≥ 0'],
                [document.getElementById('loan_amount'), v => !isNaN(v) && parseFloat(v) > 0, 'Loan amount must be positive'],
            ];

            checks.forEach(([el, test, msg]) => {
                if (!el) return;
                if (!test(el.value)) {
                    markInvalid(el, msg);
                    valid = false;
                } else {
                    markValid(el);
                }
            });

            if (!valid) {
                e.preventDefault();
                showToast('Please fix the highlighted fields.', 'danger');
            } else {
                showLoadingOverlay('Running automated risk assessment...');
            }
        });
    }

    // 4. CSV form
    const csvForm = document.getElementById('predict-csv-form');
    if (csvForm) {
        csvForm.addEventListener('submit', function (e) {
            const f = document.getElementById('csv_file');
            if (f && f.files.length === 0) {
                e.preventDefault();
                showToast('Please select a CSV file.', 'warning');
            } else {
                showLoadingOverlay('Processing batch predictions...');
            }
        });
    }

    function markInvalid(el, msg) {
        el.style.borderColor = 'var(--danger)';
        el.style.boxShadow = '0 0 0 3px rgba(255,71,87,0.15)';
        let fb = el.parentNode.querySelector('.sl-feedback');
        if (!fb) { fb = document.createElement('p'); fb.className = 'sl-feedback'; fb.style.cssText = 'color:var(--danger);font-size:0.78rem;margin:4px 0 0;'; el.parentNode.appendChild(fb); }
        fb.textContent = msg;
    }

    function markValid(el) {
        el.style.borderColor = '';
        el.style.boxShadow = '';
        const fb = el.parentNode.querySelector('.sl-feedback');
        if (fb) fb.remove();
    }

    // 5. Loading overlay
    window.showLoadingOverlay = function (msg) {
        const o = document.getElementById('loading-overlay');
        const t = document.getElementById('loading-text');
        if (t && msg) t.textContent = msg;
        if (o) o.classList.add('active');
    };

    window.hideLoadingOverlay = function () {
        const o = document.getElementById('loading-overlay');
        if (o) o.classList.remove('active');
    };

    // 6. Toast
    window.showToast = function (message, type = 'info') {
        const container = document.getElementById('toast-container-custom');
        if (!container) return;

        const icons = { success: 'fa-circle-check', danger: 'fa-triangle-exclamation', warning: 'fa-exclamation-circle', info: 'fa-circle-info' };
        const toast = document.createElement('div');
        toast.className = `sl-toast ${type}`;
        toast.innerHTML = `<i class="fas ${icons[type] || icons.info}" style="color:var(--${type === 'danger' ? 'danger' : type === 'success' ? 'accent' : type === 'warning' ? 'warning' : 'info'});flex-shrink:0;"></i><span style="flex:1;">${message}</span><button onclick="this.parentNode.remove()" style="background:none;border:none;color:var(--text-muted);cursor:pointer;padding:0 0 0 8px;font-size:1rem;">&times;</button>`;
        container.appendChild(toast);
        setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, 4500);
    };
});
