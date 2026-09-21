export async function GET() {
  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://huggingface.co/spaces/jaswanthsanjay88/mara-demo";

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Mara Research Blog</title>
    <link>${siteUrl}/research</link>
    <description>First-principles research on Atomic Function Models (AFM), 692k parameter micro-transformers, and sub-millisecond edge neural tool calling.</description>
    <language>en-us</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml" />
    <item>
      <title>Thinking in Micro-Weights: Why We Built Mara, a 692k Parameter Atomic Function Model</title>
      <link>${siteUrl}/research</link>
      <guid>${siteUrl}/research#building-mara</guid>
      <pubDate>Mon, 21 Sep 2026 00:00:00 GMT</pubDate>
      <author>jaswanthsanjay88@gmail.com (Jaswanth Sanjay)</author>
      <description><![CDATA[On the autoregressive tax, why JSON generation is an anti-pattern for embedded dispatch, and what happens when you compare a 692k parameter specialist against Needle 3 from first principles.]]></description>
    </item>
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "s-maxage=3600, stale-while-revalidate",
    },
  });
}
