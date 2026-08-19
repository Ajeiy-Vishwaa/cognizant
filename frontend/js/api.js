const API_BASE = "http://127.0.0.1:8000/api/v1";

async function loginUser(email, password) {
    const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");
    
    // Save email to localStorage upon successful login
    if (data.email) {
        localStorage.setItem("userEmail", data.email);
    }
    return data;
}

async function signupUser(email, password) {
    const res = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Signup failed");
    
    if (data.email) {
        localStorage.setItem("userEmail", data.email);
    }
    return data;
}

async function predictImage(file) {
    const email = localStorage.getItem("userEmail") || "default@example.com";
    const formData = new FormData();
    formData.append("file", file);
    formData.append("email", email);

    const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Inference error.");
    return data;
}

async function downloadReport(file) {
    const email = localStorage.getItem("userEmail") || "default@example.com";
    const formData = new FormData();
    formData.append("file", file);
    formData.append("email", email);

    const res = await fetch(`${API_BASE}/analyze-and-report`, {
        method: "POST",
        body: formData
    });

    if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to generate report.");
    }

    return await res.blob();
}