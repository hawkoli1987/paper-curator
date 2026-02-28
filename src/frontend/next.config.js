/** @type {import('next').NextConfig} */

// Backend URL: defaults to Docker service name, override with BACKEND_URL env var
// For Docker: http://backend:8000 (default)
// For Singularity/HPC: http://localhost:3100
const backendUrl = process.env.BACKEND_URL || "http://backend:8000";

const nextConfig = {
  output: "standalone",
  // Expose BACKEND_URL to API routes at runtime
  serverRuntimeConfig: {
    backendUrl: backendUrl,
  },
  // Also expose for build-time rewrites
  env: {
    BACKEND_URL: backendUrl,
  },
  async rewrites() {
    return [
      // Note: /api/arxiv/* is handled by custom API routes with extended timeout
      {
        source: "/api/pdf/:path*",
        destination: `${backendUrl}/pdf/:path*`,
      },
      {
        source: "/api/abbreviate",
        destination: `${backendUrl}/abbreviate`,
      },
      // Note: /api/qa is handled by custom API route with extended timeout
      {
        source: "/api/health",
        destination: `${backendUrl}/health`,
      },
      {
        source: "/api/config",
        destination: `${backendUrl}/config`,
      },
      {
        source: "/api/config/reset",
        destination: `${backendUrl}/config/reset`,
      },
      {
        source: "/api/ui-config",
        destination: `${backendUrl}/ui-config`,
      },
      // Database management endpoints
      {
        source: "/api/db/:path*",
        destination: `${backendUrl}/db/:path*`,
      },
      // Database & Tree endpoints
      {
        source: "/api/papers/:path*",
        destination: `${backendUrl}/papers/:path*`,
      },
      // Note: /api/tree is handled by custom API route with extended timeout
      {
        source: "/api/tree/:path*",
        destination: `${backendUrl}/tree/:path*`,
      },
      // Categories endpoints
      {
        source: "/api/categories/:path*",
        destination: `${backendUrl}/categories/:path*`,
      },
      // Topic Query endpoints
      // Note: /api/topic/:topicId/query is handled by custom API route with extended timeout
      {
        source: "/api/topic/check",
        destination: `${backendUrl}/topic/check`,
      },
      {
        source: "/api/topic/list",
        destination: `${backendUrl}/topic/list`,
      },
      {
        source: "/api/topic/search",
        destination: `${backendUrl}/topic/search`,
      },
      {
        source: "/api/topic/create",
        destination: `${backendUrl}/topic/create`,
      },
      {
        source: "/api/topic/:topicId",
        destination: `${backendUrl}/topic/:topicId`,
      },
      {
        source: "/api/topic/:topicId/papers",
        destination: `${backendUrl}/topic/:topicId/papers`,
      },
      // Feature endpoints
      {
        source: "/api/repos/:path*",
        destination: `${backendUrl}/repos/:path*`,
      },
      {
        source: "/api/references/:path*",
        destination: `${backendUrl}/references/:path*`,
      },
      // Note: /api/summarize is handled by custom API route with extended timeout
    ];
  },
};

module.exports = nextConfig;
