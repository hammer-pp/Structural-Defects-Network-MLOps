<script lang="ts">
  let file: File | null = null;
  let modelName: string = "resnet";
  let filename = "";
  let result: string | null = null;
  let imageUrl: string | null = null;
  let feedbackMessage = "";

  async function handleSubmit() {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelName);

    const response = await fetch("http:localhost:8000/", {
      method: "POST",
      body: formData
    });

    const html = await response.text();
    document.open();
    document.write(html);
    document.close();
  }

  function onFileChange(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      file = input.files[0];
      filename = file.name;
    }
  }
</script>

<div class="min-h-screen bg-gray-100 flex items-center justify-center p-6">
  <div class="bg-white shadow-lg rounded-lg p-8 w-full max-w-2xl">
    <h1 class="text-2xl font-bold text-center text-gray-800 mb-4">Surface Crack Detector</h1>
    <p class="text-gray-600 text-center mb-6">Upload an image of a concrete surface to detect cracks.</p>

    <form on:submit|preventDefault={handleSubmit} class="space-y-4">
      <div class="flex items-center justify-center w-full px-4 py-10 border-2 border-dashed border-gray-400 rounded-lg cursor-pointer bg-gray-50 hover:border-blue-500 transition">
        <input type="file" accept="image/*" on:change={onFileChange} class="hidden" id="fileInput" />
        <label for="fileInput" class="text-gray-600 text-center cursor-pointer">
          Drag & Drop or <span class="text-blue-600 underline">click to upload</span>
        </label>
      </div>
      {#if filename}
        <p class="text-sm text-gray-500 text-center">📄 {filename} selected.</p>
      {/if}

      <label for="modelSelect" class="block mb-1 font-medium">Choose Model:</label>
      <select bind:value={modelName} id="modelSelect" class="w-full border-gray-300 rounded p-2">
        <option value="resnet">ResNet</option>
        <option value="mobilenet">MobileNet</option>
        <option value="custom">Custom CNN</option>
      </select>

      <div class="text-center">
        <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">Predict</button>
      </div>
    </form>
  </div>
</div>

<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
  :global(body) {
    font-family: 'Inter', sans-serif;
  }
</style>
