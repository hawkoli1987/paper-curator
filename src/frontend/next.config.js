/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      // Core endpoints
      {
        source: "/api/arxiv/:path*",
        destination: "http://backend:8000/arxiv/:path*",
      },
      {
        source: "/api/pdf/:path*",
        destination: "http://backend:8000/pdf/:path*",
      },
      {
        source: "/api/classify",
        destination: "http://backend:8000/classify",
      },
      {
        source: "/api/embed",
        destination: "http://backend:8000/embed",
      },
      {
        source: "/api/qa",
        destination: "http://backend:8000/qa",
      },
      {
        source: "/api/health",
        destination: "http://backend:8000/health",
      },
      {
        source: "/api/config",
        destination: "http://backend:8000/config",
      },
      // Database & Tree endpoints
      {
        source: "/api/papers/:path*",
        destination: "http://backend:8000/papers/:path*",
      },
      {
        source: "/api/tree",
        destination: "http://backend:8000/tree",
      },
      {
        source: "/api/tree/:path*",
        destination: "http://backend:8000/tree/:path*",
      },
      // Feature endpoints
      {
        source: "/api/repos/:path*",
        destination: "http://backend:8000/repos/:path*",
      },
      {
        source: "/api/references/:path*",
        destination: "http://backend:8000/references/:path*",
      },
      // Note: /api/summarize is handled by custom API route with extended timeout
    ];
  },
};

module.exports = nextConfig;
