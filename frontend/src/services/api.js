const API_URL = "http://127.0.0.1:8000";

export async function predictSign(
  image,
  modelType,
  sessionId
) {
  const response = await fetch(
    `${API_URL}/predict`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        image,
        model_type: modelType,
        session_id: sessionId,
      }),
    }
  );

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result.error ||
        "Prediksi gagal diproses oleh backend."
    );
  }

  return result;
}

export async function resetWordSequence(
  sessionId
) {
  const response = await fetch(
    `${API_URL}/reset-sequence`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        session_id: sessionId,
      }),
    }
  );

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result.error ||
        "Buffer sequence gagal dibersihkan."
    );
  }

  return result;
}