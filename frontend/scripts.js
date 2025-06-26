/**
 * Vietnamese Text Similarity Frontend
 * Main JavaScript file for handling UI interactions and API calls
 */

class TextSimilarityApp {
  constructor() {
    // Cập nhật port API thành 5001
    this.apiBaseUrls = [
      "http://127.0.0.1:5001/api",
      "http://localhost:5001/api",
      "http://0.0.0.0:5001/api",
    ];
    this.apiBaseUrl = this.apiBaseUrls[0];
    this.currentMode = "manual";
    this.datasets = [];
    this.comparisonChart = null;
    this.categories = [];
    this.labels = [];
    this.init();
  }

  async init() {
    console.log("🚀 Initializing TextSimilarityApp...");

    // Setup UI first
    this.setupEventListeners();

    // Try to find working API URL
    await this.findWorkingApiUrl();

    // Check backend connection
    const backendHealthy = await this.checkBackendHealth();

    if (backendHealthy) {
      await this.loadCategories();
      await this.loadLabels();
      await this.loadDatasets();
      await this.updateTokenizerOptions();
      await this.loadMethodsInfo();
      console.log("✓ Backend connection established");
    } else {
      console.warn("⚠ Backend connection failed, using fallback data");
      this.datasets = this.getFallbackDatasets();
      this.populateDatasetSelect();
      this.setDefaultTokenizerOptions();
      this.populateMethodsWithDefaults();
    }

    console.log("✓ TextSimilarityApp initialized");
  }

  async findWorkingApiUrl() {
    for (const url of this.apiBaseUrls) {
      try {
        const response = await fetch(`${url}/health`, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        });
        if (response.ok) {
          this.apiBaseUrl = url;
          console.log(`✓ Found working API at: ${url}`);
          return;
        }
      } catch (error) {
        console.log(`✗ Failed to connect to: ${url}`);
      }
    }
    console.warn("⚠ No working API URL found, using default");
  }

  async checkBackendHealth() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error("Backend health check failed:", error);
      return false;
    }
  }

  setupEventListeners() {
    // Input mode toggle
    const manualBtn = document.getElementById("manual-input-btn");
    const datasetBtn = document.getElementById("dataset-input-btn");

    if (manualBtn) {
      manualBtn.addEventListener("click", () => this.switchInputMode("manual"));
    }
    if (datasetBtn) {
      datasetBtn.addEventListener("click", () =>
        this.switchInputMode("dataset")
      );
    }

    // Text input character counting
    const text1 = document.getElementById("text1");
    const text2 = document.getElementById("text2");

    if (text1) {
      text1.addEventListener("input", (e) => {
        this.updateCharCount("char-count-1", e.target.value.length);
      });
    }
    if (text2) {
      text2.addEventListener("input", (e) => {
        this.updateCharCount("char-count-2", e.target.value.length);
      });
    }

    // Dataset selection
    const datasetSelect = document.getElementById("dataset-select");
    if (datasetSelect) {
      datasetSelect.addEventListener("change", (e) => {
        const index = parseInt(e.target.value);
        if (!isNaN(index) && this.datasets[index]) {
          const item = this.datasets[index];

          // Update preview
          const preview1 = document.getElementById("preview-text1");
          const preview2 = document.getElementById("preview-text2");

          if (preview1) preview1.textContent = item.text1;
          if (preview2) preview2.textContent = item.text2;

          // Show preview section
          const previewSection = document.getElementById("dataset-preview");
          if (previewSection) previewSection.style.display = "block";
        } else {
          // Hide preview section if no valid selection
          const previewSection = document.getElementById("dataset-preview");
          if (previewSection) previewSection.style.display = "none";
        }
      });
    }

    // Calculate button
    const calculateBtn = document.getElementById("calculate-btn");
    if (calculateBtn) {
      calculateBtn.addEventListener("click", () => this.calculateSimilarity());
    }

    // Compare all methods button
    const compareBtn = document.getElementById("compare-all-btn");
    if (compareBtn) {
      compareBtn.addEventListener("click", () => this.compareAllMethods());
    }

    // Export results button
    const exportBtn = document.getElementById("export-btn");
    if (exportBtn) {
      exportBtn.addEventListener("click", () => this.exportResults());
    }

    // Add filter change handlers
    const categorySelect = document.getElementById("category-select");
    const labelSelect = document.getElementById("label-select");

    if (categorySelect) {
      categorySelect.addEventListener("change", () =>
        this.loadFilteredDatasets()
      );
    }
    if (labelSelect) {
      labelSelect.addEventListener("change", () => this.loadFilteredDatasets());
    }
  }

  switchInputMode(mode) {
    this.currentMode = mode;

    // Update button states
    document.querySelectorAll(".toggle-btn").forEach((btn) => {
      btn.classList.remove("active");
    });

    const activeBtn = document.getElementById(`${mode}-input-btn`);
    if (activeBtn) {
      activeBtn.classList.add("active");
    }

    // Show/hide input modes
    document.querySelectorAll(".input-mode").forEach((modeEl) => {
      modeEl.style.display = "none";
    });

    const activeMode = document.getElementById(`${mode}-input`);
    if (activeMode) {
      activeMode.style.display = "block";
    }
  }

  updateCharCount(elementId, count) {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = `${count} ký tự`;

      if (count > 5000) {
        element.style.color = "#dc3545"; // Red for very long text
      } else if (count > 1000) {
        element.style.color = "#ffc107"; // Yellow for moderately long text
      } else {
        element.style.color = "#666"; // Default color
      }
    }
  }

  async loadDatasets() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/dataset`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      if (result.status === "success" && Array.isArray(result.data)) {
        this.datasets = result.data;
        this.populateDatasetSelect();
        console.log("✓ Datasets loaded successfully");
      } else {
        throw new Error("Invalid dataset format");
      }
    } catch (error) {
      console.error("Error loading datasets:", error);
      this.showNotification(
        "Không thể tải dataset, sử dụng dữ liệu mẫu",
        "warning"
      );
      this.datasets = this.getFallbackDatasets();
      this.populateDatasetSelect();
    }
  }

  populateDatasetSelect() {
    const datasetSelect = document.getElementById("dataset-select");
    if (!datasetSelect) return;

    // Clear existing options
    datasetSelect.innerHTML =
      '<option value="">-- Chọn cặp văn bản --</option>';

    // Add new options
    this.datasets.forEach((dataset, index) => {
      const option = document.createElement("option");
      option.value = index;
      // Tạo preview ngắn gọn cho option
      const text1Preview =
        dataset.text1.substring(0, 30) +
        (dataset.text1.length > 30 ? "..." : "");
      const text2Preview =
        dataset.text2.substring(0, 30) +
        (dataset.text2.length > 30 ? "..." : "");
      option.textContent = `Cặp ${
        index + 1
      }: ${text1Preview} | ${text2Preview}`;
      datasetSelect.appendChild(option);
    });
  }

  getFallbackDatasets() {
    return [
      {
        text1: "Hà Nội là thủ đô của Việt Nam",
        text2: "Thủ đô của Việt Nam là Hà Nội",
      },
      {
        text1: "Học sinh đến trường",
        text2: "Học sinh tới trường học",
      },
    ];
  }

  async calculateSimilarity() {
    try {
      this.showLoading(true, "Đang tính toán độ tương đồng...");

      const texts = this.getInputTexts();
      if (!texts || !this.validateInputs(texts.text1, texts.text2)) {
        return;
      }

      const tokenizer = document.getElementById("tokenizer").value;
      const method = document.getElementById("similarity-select").value;

      const response = await fetch(`${this.apiBaseUrl}/similarity`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text1: texts.text1,
          text2: texts.text2,
          similarity_method: method,
          tokenize_method: tokenizer,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Hiển thị kết quả
      this.displaySingleResult({
        similarity: result.similarity_score,
        method: result.similarity_method,
        tokenizer: result.tokenize_method,
        execution_time: result.execution_time || 0,
        tokens_count_1: result.text1_length,
        tokens_count_2: result.text2_length,
      });
    } catch (error) {
      console.error("Error calculating similarity:", error);
      this.showNotification(
        "Có lỗi xảy ra khi tính toán: " + error.message,
        "error"
      );
    } finally {
      this.showLoading(false);
    }
  }

  async compareAllMethods() {
    try {
      this.showLoading(true, "Đang so sánh các phương pháp...");

      const texts = this.getInputTexts();
      if (!texts || !this.validateInputs(texts.text1, texts.text2)) {
        return;
      }

      const response = await fetch(`${this.apiBaseUrl}/compare_methods`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text1: texts.text1,
          text2: texts.text2,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      this.displayComparisonResults(results);
    } catch (error) {
      console.error("Error comparing methods:", error);
      this.showNotification(
        "Có lỗi xảy ra khi so sánh: " + error.message,
        "error"
      );
    } finally {
      this.showLoading(false);
    }
  }

  displaySingleResult(result) {
    // Show results section
    const resultsSection = document.getElementById("results-section");
    const singleResult = document.getElementById("single-result");
    const comparisonResult = document.getElementById("comparison-result");

    if (resultsSection) resultsSection.style.display = "block";
    if (singleResult) singleResult.style.display = "block";
    if (comparisonResult) comparisonResult.style.display = "none";

    // Update score
    const scoreElement = document.getElementById("similarity-score");
    if (scoreElement) {
      const score = (result.similarity * 100).toFixed(2);
      scoreElement.textContent = `${score}%`;
    }

    // Update details
    const methodElement = document.getElementById("method-used");
    if (methodElement) methodElement.textContent = result.method;

    const tokenizerElement = document.getElementById("tokenizer-used");
    if (tokenizerElement) tokenizerElement.textContent = result.tokenizer;

    const executionElement = document.getElementById("execution-time");
    if (executionElement)
      executionElement.textContent = `${result.execution_time.toFixed(3)}s`;

    const tokens1Element = document.getElementById("tokens-count-1");
    if (tokens1Element) tokens1Element.textContent = result.tokens_count_1;

    const tokens2Element = document.getElementById("tokens-count-2");
    if (tokens2Element) tokens2Element.textContent = result.tokens_count_2;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  displayComparisonResults(results) {
    // Show results section
    const resultsSection = document.getElementById("results-section");
    const singleResult = document.getElementById("single-result");
    const comparisonResult = document.getElementById("comparison-result");

    if (resultsSection) resultsSection.style.display = "block";
    if (singleResult) singleResult.style.display = "none";
    if (comparisonResult) comparisonResult.style.display = "block";

    // Create chart
    this.createComparisonChart(results);

    // Create table
    this.populateComparisonTable(results);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  getInputTexts() {
    if (this.currentMode === "manual") {
      return {
        text1: document.getElementById("text1").value.trim(),
        text2: document.getElementById("text2").value.trim(),
      };
    } else {
      const datasetSelect = document.getElementById("dataset-select");
      const selectedIndex = parseInt(datasetSelect.value);

      if (isNaN(selectedIndex) || !this.datasets[selectedIndex]) {
        this.showNotification(
          "Vui lòng chọn một cặp văn bản từ dataset",
          "error"
        );
        return null;
      }

      const selectedData = this.datasets[selectedIndex];
      return {
        text1: selectedData.text1,
        text2: selectedData.text2,
      };
    }
  }

  validateInputs(text1, text2) {
    if (!text1 || !text2) {
      this.showNotification("Vui lòng nhập đầy đủ cả hai văn bản", "error");
      return false;
    }

    if (text1.length > 10000 || text2.length > 10000) {
      this.showNotification("Văn bản quá dài (giới hạn 10000 ký tự)", "error");
      return false;
    }

    return true;
  }

  showLoading(show, message = "Đang xử lý...") {
    const loadingElement = document.getElementById("loading");
    const loadingMessage = document.getElementById("loading-message");

    if (loadingElement) {
      loadingElement.style.display = show ? "flex" : "none";
    }

    if (loadingMessage) {
      loadingMessage.textContent = message;
    }
  }

  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.innerHTML = `
            <div class="notification-icon">${this.getNotificationIcon(
              type
            )}</div>
            <div class="notification-message">${message}</div>
        `;

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      notification.remove();
    }, 5000);
  }

  getNotificationIcon(type) {
    switch (type) {
      case "success":
        return "✓";
      case "error":
        return "✗";
      case "warning":
        return "⚠";
      default:
        return "ℹ";
    }
  }

  exportResults() {
    // Implementation for exporting results
    this.showNotification(
      "Tính năng xuất kết quả đang được phát triển",
      "info"
    );
  }

  async loadCategories() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/categories`);
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const result = await response.json();
      if (result.status === "success" && Array.isArray(result.data)) {
        this.categories = result.data;
        this.populateCategorySelect();
      }
    } catch (error) {
      console.error("Failed to load categories:", error);
    }
  }

  async loadLabels() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/labels`);
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const result = await response.json();
      if (result.status === "success" && Array.isArray(result.data)) {
        this.labels = result.data;
        this.populateLabelSelect();
      }
    } catch (error) {
      console.error("Failed to load labels:", error);
    }
  }

  populateCategorySelect() {
    const select = document.getElementById("category-select");
    if (!select) return;

    select.innerHTML = '<option value="">Tất cả danh mục</option>';
    this.categories.forEach((category) => {
      const option = document.createElement("option");
      option.value = category;
      option.textContent = category;
      select.appendChild(option);
    });
  }

  populateLabelSelect() {
    const select = document.getElementById("label-select");
    if (!select) return;

    select.innerHTML = '<option value="">Tất cả nhãn</option>';
    this.labels.forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      select.appendChild(option);
    });
  }

  async loadFilteredDatasets() {
    const category = document.getElementById("category-select")?.value || "";
    const label = document.getElementById("label-select")?.value || "";

    try {
      const queryParams = new URLSearchParams();
      if (category) queryParams.append("category", category);
      if (label) queryParams.append("label", label);

      const response = await fetch(`${this.apiBaseUrl}/dataset?${queryParams}`);
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const result = await response.json();
      if (result.status === "success" && Array.isArray(result.data)) {
        this.datasets = result.data;
        this.populateDatasetSelect();
      }
    } catch (error) {
      console.error("Failed to load filtered datasets:", error);
      this.showNotification("Không thể tải dữ liệu đã lọc", "error");
    }
  }

  // Update existing dataset preview function
  updateDatasetPreview(item) {
    const preview1 = document.getElementById("preview-text1");
    const preview2 = document.getElementById("preview-text2");
    const previewCategory = document.getElementById("preview-category");
    const previewLabel = document.getElementById("preview-label");

    if (preview1) preview1.textContent = item.text1;
    if (preview2) preview2.textContent = item.text2;
    if (previewCategory) previewCategory.textContent = item.category || "N/A";
    if (previewLabel) previewLabel.textContent = item.label || "N/A";

    const previewSection = document.getElementById("dataset-preview");
    if (previewSection) previewSection.style.display = "block";
  }
}

// Initialize the app when the page loads
document.addEventListener("DOMContentLoaded", () => {
  window.app = new TextSimilarityApp();
});
