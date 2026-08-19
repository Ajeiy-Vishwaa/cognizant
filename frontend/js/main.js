const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewSection = document.getElementById("previewSection");
const imagePreview = document.getElementById("imagePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("resultSection");
const statusBadge = document.getElementById("statusBadge");
const scoreValue = document.getElementById("scoreValue");

const damageRow = document.getElementById("damageRow");
const damageValue = document.getElementById("damageValue");
const boundingContainer = document.getElementById("boundingContainer");
const annotatedPreview = document.getElementById("annotatedPreview");
const gradcamContainer = document.getElementById("gradcamContainer");
const gradcamPreview = document.getElementById("gradcamPreview");
const downloadReportBtn = document.getElementById("downloadReportBtn");

const authForm = document.getElementById("authForm");
const authError = document.getElementById("authError");
const authView = document.getElementById("authView");
const dashboardView = document.getElementById("dashboardView");
const userNav = document.getElementById("userNav");
const userEmailDisplay = document.getElementById("userEmailDisplay");
const logoutBtn = document.getElementById("logoutBtn");

const tabLogin = document.getElementById("tabLogin");
const tabSignup = document.getElementById("tabSignup");
const authTitle = document.getElementById("authTitle");
const authSub = document.getElementById("authSub");
const authSubmitBtn = document.getElementById("authSubmitBtn");

let currentFile = null;

// --- Auth Tab Switchers ---
if (tabLogin && tabSignup) {
    tabLogin.addEventListener("click", () => {
        tabLogin.classList.add("active");
        tabSignup.classList.remove("active");
        authTitle.textContent = "Welcome Back";
        authSub.textContent = "Sign in to access damage & fraud analysis";
        authSubmitBtn.textContent = "Log In";
        if (authError) authError.style.display = "none";
    });

    tabSignup.addEventListener("click", () => {
        tabSignup.classList.add("active");
        tabLogin.classList.remove("active");
        authTitle.textContent = "Create an Account";
        authSub.textContent = "Sign up to start evaluating vehicle claims";
        authSubmitBtn.textContent = "Sign Up";
        if (authError) authError.style.display = "none";
    });
}

// --- Authentication Handler ---
if (authForm) {
    authForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const emailInput = document.getElementById("email");
        const passwordInput = document.getElementById("password");

        if (!emailInput || !passwordInput) return;

        const email = emailInput.value;
        const password = passwordInput.value;
        const isSignup = authTitle ? authTitle.textContent.includes("Account") : false;

        if (authError) authError.style.display = "none";
        if (authSubmitBtn) authSubmitBtn.disabled = true;

        try {
            if (isSignup) {
                await signupUser(email, password);
            } else {
                await loginUser(email, password);
            }

            if (authView) authView.style.display = "none";
            if (dashboardView) dashboardView.style.display = "block";
            if (userNav) userNav.style.display = "flex";
            if (userEmailDisplay) userEmailDisplay.textContent = email;

        } catch (err) {
            if (authError) {
                authError.textContent = err.message || "Authentication failed.";
                authError.style.display = "block";
            }
        } finally {
            if (authSubmitBtn) authSubmitBtn.disabled = false;
        }
    });
}

// --- Logout Handler ---
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("userEmail");
        if (authView) authView.style.display = "block";
        if (dashboardView) dashboardView.style.display = "none";
        if (userNav) userNav.style.display = "none";
        
        const emailInput = document.getElementById("email");
        const passwordInput = document.getElementById("password");
        if (emailInput) emailInput.value = "";
        if (passwordInput) passwordInput.value = "";
        
        if (previewSection) previewSection.style.display = "none";
        if (resultSection) resultSection.style.display = "none";
        currentFile = null;
    });
}

// --- Drag & Drop / File Selection ---
if (dropZone) dropZone.addEventListener("click", () => fileInput && fileInput.click());

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) processFile(e.target.files[0]);
    });
}

function processFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        if (imagePreview) imagePreview.src = e.target.result;
        if (previewSection) previewSection.style.display = "block";
        if (resultSection) resultSection.style.display = "none";
    };
    reader.readAsDataURL(file);
}

// --- Image Analysis Handler ---
if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) {
            alert("Please select or drop an image file first.");
            return;
        }

        analyzeBtn.textContent = "Processing Image...";
        analyzeBtn.disabled = true;

        try {
            const data = await predictImage(currentFile);
            const toImageSource = (image) => image && (image.startsWith("data:")
                ? image
                : `data:image/jpeg;base64,${image}`);

            if (statusBadge) statusBadge.textContent = data.assessment || "Evaluated";
            if (scoreValue) scoreValue.textContent = `${data.fraud_risk_score.toFixed(2)}%`;

            if (data.damage_analysis && damageRow && damageValue) {
                const severityLabel = data.damage_analysis.overall_severity || "N/A";
                const severityScore = data.damage_analysis.overall_severity_score ?? 0;
                damageValue.textContent = `${severityLabel} (${severityScore}/100)`;
                damageRow.style.display = "block";
            }

            const annotatedImage = toImageSource(data.annotated_image || data.annotated_image_url);
            if (boundingContainer && annotatedPreview && annotatedImage) {
                annotatedPreview.src = annotatedImage;
                boundingContainer.style.display = "block";
            }

            const gradcamImage = toImageSource(data.gradcam_image || data.gradcam_image_base64);
            if (gradcamContainer && gradcamPreview && gradcamImage) {
                gradcamPreview.src = gradcamImage;
                gradcamContainer.style.display = "block";
            }

            if (resultSection) resultSection.style.display = "block";

        } catch (err) {
            alert(err.message || "Inference error. Ensure your FastAPI server is running on port 8000.");
            console.error(err);
        } finally {
            analyzeBtn.textContent = "Analyze Claim Image";
            analyzeBtn.disabled = false;
        }
    });
}

// --- Report Download Handler ---
if (downloadReportBtn) {
    downloadReportBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        try {
            const blob = await downloadReport(currentFile);
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `claim_report_${currentFile.name}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            alert(err.message || "Error generating PDF report.");
            console.error(err);
        }
    });
}