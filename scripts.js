document.addEventListener("DOMContentLoaded", () => {
    const header = document.getElementById("site-header");
    const themeToggle = document.getElementById("theme-toggle");
    const projectsContainer = document.getElementById("projects-container");
    const articlesContainer = document.getElementById("articles-container");

    // Apply saved theme
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") {
        document.body.classList.add("light-mode");
        themeToggle.textContent = "☀️";
    } else {
        themeToggle.textContent = "🌙";
    }

    // Toggle theme button (just once!)
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

    // Sticky header shadow
    window.addEventListener("scroll", () => {
        if (window.scrollY > 50) {
            header.classList.add("scrolled");
        } else {
            header.classList.remove("scrolled");
        }
    });

    // Load featured content
    fetch("projects/project_list.json")
        .then(res => res.json())
        .then(projects => {
            const container = document.getElementById("recent-projects-container");
            displayFeaturedProjects(projects, container);
        });

    fetch("blog/post_list.json")
        .then(response => response.json())
        .then(data => displayFeaturedItems(data, articlesContainer));
});

function displayFeaturedProjects(projects, container) {
    // Filter for featured projects only
    // const featured = projects.filter(p => p.categories && p.categories.includes("Featured"));
  
    // Sort by date (newest first)
    projects.sort((a, b) => new Date(b.created_date) - new Date(a.created_date));
  
    // Take top 3
    projects.slice(0, 3).forEach(project => {
      const item = document.createElement("div");
      const formattedDate = new Date(project.created_date).toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "long",
        year: "numeric"
      });
      item.className = "featured-item";
      item.innerHTML = `
        <img src="projects/${project.image}" alt="${project.title}">
        <h3>${project.title}</h3>
        <p class="project-date">${formattedDate}</p>
        <p>${project.description}</p>
        <a href="${project.link}" class="btn" >Explore</a>
      `;
      container.appendChild(item);
    });
  }

function displayFeaturedItems(data, container, type) {
    const featuredItems = data.filter(item => item.categories.includes("Featured"));
  
    featuredItems.forEach(item => {
      const itemDiv = document.createElement("div");
      itemDiv.className = "featured-item";
      itemDiv.innerHTML = `
        <img src="blog/${item.image}" alt="${item.title}">
        <h3>${item.title}</h3>
        <p>${item.published_at}</p>
        <a href="post/main.html?post=${encodeURIComponent(item.title)}" class="btn">Read more...</a>
      `;
      container.appendChild(itemDiv);
    });
  }
  
