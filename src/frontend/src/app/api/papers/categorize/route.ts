import { NextRequest, NextResponse } from "next/server";
import { backendPost } from "@/lib/backend-proxy";

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const full = searchParams.get("full") === "true";

  const backendPath = full
    ? "/papers/categorize?full=true"
    : "/papers/categorize";

  // Full rebuild can take 30+ minutes; partial is much faster but may still
  // take several minutes for large libraries.
  const timeoutMs = full ? 1_800_000 : 900_000;

  try {
    const { status, body } = await backendPost(backendPath, { timeoutMs });

    try {
      const data = JSON.parse(body);
      return NextResponse.json(data, { status });
    } catch {
      return NextResponse.json(
        { error: "Invalid response from backend", details: body.slice(0, 500) },
        { status: 502 }
      );
    }
  } catch (error) {
    console.error("Categorize request failed:", error);
    return NextResponse.json(
      { error: "Categorize request failed", details: String(error) },
      { status: 504 }
    );
  }
}
