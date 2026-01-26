/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      // Exclude /api/summarize - handled by custom API route with extended timeout
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
    ];
  },
};

module.exports = nextConfig;
