document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const currentTheme = localStorage.getItem("theme");
  
    if (currentTheme === "light") {
      document.body.classList.add("light-mode");
      themeToggle.textContent = "☀️";
    } else {
      themeToggle.textContent = "🌙";
    }
  
    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("light-mode");
  
      if (document.body.classList.contains("light-mode")) {
        localStorage.setItem("theme", "light");
        themeToggle.textContent = "☀️";
      } else {
        localStorage.setItem("theme", "dark");
        themeToggle.textContent = "🌙";
      }
    });
  });
  