export async function GET() {
  try {
    const backendRes = await fetch("http://127.0.0.1:8000/api/health", {
      cache: "no-store",
    });
    if (!backendRes.ok) {
      return Response.json({ status: "error" }, { status: backendRes.status });
    }
    const data = await backendRes.json();
    return Response.json(data);
  } catch (error) {
    return Response.json(
      { status: "offline", error: error.message },
      { status: 502 }
    );
  }
}
