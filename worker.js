const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,HEAD,POST,PUT,PATCH,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Expose-Headers": "Content-Disposition, Content-Type",
};

function withCors(response) {
  const headers = new Headers(response.headers);
  Object.entries(corsHeaders).forEach(([name, value]) => headers.set(name, value));
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const backendUrl = env.BACKEND_URL;
    if (!backendUrl) {
      return withCors(
        Response.json(
          { detail: "BACKEND_URL is not configured for this Worker." },
          { status: 500 },
        ),
      );
    }

    const incomingUrl = new URL(request.url);
    const targetUrl = new URL(incomingUrl.pathname + incomingUrl.search, backendUrl);
    const headers = new Headers(request.headers);
    headers.delete("host");

    try {
      const response = await fetch(targetUrl, {
        method: request.method,
        headers,
        body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
        redirect: "manual",
      });
      return withCors(response);
    } catch (error) {
      return withCors(
        Response.json(
          { detail: "Unable to reach the analysis backend." },
          { status: 502 },
        ),
      );
    }
  },
};
