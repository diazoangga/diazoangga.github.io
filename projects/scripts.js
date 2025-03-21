document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("projects-container");
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
  
    fetch("project_list.json")
      .then(res => res.json())
      .then(data => {
        data.sort((a, b) => new Date(b.created_date) - new Date(a.created_date));

        data.forEach(project => {
        const item = document.createElement("div");
        const formattedDate = new Date(project.created_date).toLocaleDateString("en-GB", {
            month: "long",
            day: "2-digit",
            year: "numeric"
          });
          
        item.className = "project-item";
        item.innerHTML = `
            <img src="${project.image}" alt="${project.title}" />
            <h3>${project.title}</h3>
            <p class="project-date">${formattedDate}</p>
            <p>${project.description}</p>
            <a href="${project.link}" class="btn" target="_blank" rel="noopener">Explore</a>
        `;
        container.appendChild(item);
        });
      });
  });
  