export async function POST() {
  try {
    const backendRes = await fetch("http://127.0.0.1:8000/api/reset", {
      method: "POST",
      cache: "no-store",
    });
    if (!backendRes.ok) {
      return Response.json({ error: "Failed to reset state" }, { status: backendRes.status });
    }
    const data = await backendRes.json();
    return Response.json(data);
  } catch (error) {
    return Response.json(
      { error: "Backend unreachable", message: error.message },
      { status: 502 }
    );
  }
}
