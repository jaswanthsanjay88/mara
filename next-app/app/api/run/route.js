export async function POST(request) {
  try {
    const body = await request.json();
    const backendRes = await fetch("http://127.0.0.1:8000/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    if (!backendRes.ok) {
      const errText = await backendRes.text();
      return Response.json(
        { error: "Backend error", details: errText },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    return Response.json(data);
  } catch (error) {
    return Response.json(
      {
        error: "Failed to connect to Mara AFM inference daemon",
        message: error.message,
      },
      { status: 502 }
    );
  }
}
