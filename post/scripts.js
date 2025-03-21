function slugify(str) {
    return str.replace(/[^a-zA-Z0-9 ]/g, '') // remove special chars
              .replace(/\s+/g, ' ')         // normalize spacing
              .trim();
  }

  
document.addEventListener("DOMContentLoaded", () => {
    const postTitle = document.getElementById("post-title");
    const postDate = document.getElementById("post-date");
    const postCategories = document.getElementById("post-categories");
    const postContent = document.getElementById("post-content");
    const postContainer = document.getElementById("post-container");
  
    const urlParams = new URLSearchParams(window.location.search);
    const postFilename = decodeURIComponent(urlParams.get("post"));

    const themeToggle = document.getElementById("theme-toggle");
    const currentTheme = localStorage.getItem("theme");

    // Apply saved theme on page load
    if (currentTheme === "light") {
        document.body.classList.add("light-mode");
        themeToggle.textContent = "☀️"; // Sun icon for light mode
    } else {
        themeToggle.textContent = "🌙"; // Moon icon for dark mode
    }

    // Toggle theme on button click
    themeToggle.addEventListener("click", () => {
        document.body.classList.toggle("light-mode");

        if (document.body.classList.contains("light-mode")) {
            localStorage.setItem("theme", "light");
            themeToggle.textContent = "☀️"; // Switch to sun when in light mode
        } else {
            localStorage.setItem("theme", "dark");
            themeToggle.textContent = "🌙"; // Switch to moon when in dark mode
        }
    });

    const postSlug = slugify(postFilename);
        
    const mdPath = `../blog/assets/${postSlug}/content.md`;
    const basePath = `../blog/assets/${postSlug}/`;
  
    // Step 1: Load post metadata from post_list.json
    fetch("../blog/post_list.json")
      .then(res => res.json())
      .then(posts => {
        const matchedPost = posts.find(post => post.title === postFilename);
  
        if (!matchedPost) {
          throw new Error("Post not found in post_list.json");
        }
  
        // ✅ Populate metadata
        postTitle.textContent = matchedPost.title;
        postDate.textContent = `Published on: ${matchedPost.published_at}`;
  
        postCategories.innerHTML = matchedPost.categories.map(cat => {
          const className = `badge ${getCategoryClass(cat)}`;
          return `<span class="${className}">${cat}</span>`;
        }).join(" ");
  
        // ✅ Step 2: Load Markdown content (no metadata in the file)
        // const mdPath = `../blog/assets/${postFilename}/content.md`;
        

  
        return fetch(mdPath);
      })
      .then(res => {
        if (!res.ok) throw new Error("Markdown not found");
        return res.text();
      })
      .then(data => {
        // console.log('ok')
        // Fix $$ block equations wrapping
        const fixedMarkdown = data.replace(
          /^\$\$(.*?)\$\$/gms,
          (_, eqn) => `<div>$$${eqn}$$</div>`
        );
  
        marked.setOptions({
          mangle: false,
          headerIds: false,
          breaks: true
        });
  
        const contentOnly = fixedMarkdown
            .split('\n')
            .filter(line =>
                !line.startsWith('Published Date:') &&
                !line.startsWith('Tags:') &&
                !line.startsWith('# ')
            )
            .join('\n')
            .trim();

        // console.log(contentOnly)
        // const basePath = `../blog/assets/${postFilename}/`;
        // console.log(str(mdPath))
        

        // console.log(contentOnly)
        const lines = contentOnly.split('\n');
        let outputLines = [];
        

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            const match = line.match(/^!\[([^\]]+)\]\(([^)]+)\)$/);
            

            if (match) {
                const alt = match[1].trim();
                const imgPath = match[2].trim();

                // Create figure element
                outputLines.push(
                `<figure><img src="${basePath}${imgPath}" alt="${alt}"><figcaption>${alt}</figcaption></figure>`
                );

                // Check if next line is an exact match of the alt text and skip it
                if (lines[i + 1] && lines[i + 1].trim() === alt) {
                i++; // skip next line
                }
            } else {
                outputLines.push(lines[i]);
            }
        }

        

        const finalMarkdown = outputLines.join('\n');
        postContent.innerHTML = marked.parse(finalMarkdown);

        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
          });

        // wrap in .post-content div to ensure styles apply
        // const renderedHTML = `<div class="post-content">${marked.parse(finalMarkdown)}</div>`;
        // postContent.innerHTML = renderedHTML;


  
        renderMathInElement(postContent, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false }
          ],
          throwOnError: false
        });
      })
      .catch(error => {
        postContainer.innerHTML = "<p>Error loading post.</p>";
        console.error("Error loading post:", error);
      });
  });
  
  // Helper to assign badge class
  function getCategoryClass(category) {
    const normalized = category.trim().toLowerCase();
  
    const map = {
      "machine learning": "badge-ml",
      "computer vision": "badge-cv",
      "data analytics": "badge-da",
      "featured": "badge-fe"
    };
  
    return map[normalized] || "badge-default";
  }
  
  