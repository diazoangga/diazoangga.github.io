document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const mlPostsContainer = document.getElementById("ml-posts");
    const cvPostsContainer = document.getElementById("cv-posts");
    const gaPostsContainer = document.getElementById("ga-posts");
    const daPostsContainer = document.getElementById("da-posts");
  
    // Apply saved theme
    if (localStorage.getItem("theme") === "light") {
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
  
    fetch("post_list.json")
      .then(response => response.json())
      .then(data => {
        populateBlogPosts(data, mlPostsContainer, "Machine Learning");
        populateBlogPosts(data, cvPostsContainer, "Computer Vision");
        populateBlogPosts(data, daPostsContainer, "Data Analytics");
        populateBlogPosts(data, gaPostsContainer, "Generative AI");
      });
  });
  
  function populateBlogPosts(posts, container, category) {
    posts.forEach(post => {
      if (post.categories.includes(category)) {
        const postElement = document.createElement("div");
        postElement.classList.add("blog-post");
        postElement.innerHTML = `
          <a href="../post/main.html?post=${encodeURIComponent(post.title)}">
            <img src="./${post.image}" alt="${post.title}">
          </a>
          <h3><a href="../post/main.html?post=${encodeURIComponent(post.title)}">${post.title}</a></h3>
          <p>${post.published_at}</p>
          <p>${post.categories.map(cat => `<span class='badge ${getCategoryClass(cat)}'>${cat}</span>`).join(' ')}</p>
          <a href="../post/main.html?post=${encodeURIComponent(post.title)}" class="read-more">Read More</a>
        `;
        container.appendChild(postElement);
      }
    });
  }
  
  function getCategoryClass(category) {
    const categoryMap = {
      "Machine Learning": "badge-ml",
      "Computer Vision": "badge-cv",
      "Data Analytics": "badge-da",
      "Generative AI": "badge-ga",
      "Featured": "badge-featured"
    };
    return categoryMap[category] || "badge-default";
  }
  