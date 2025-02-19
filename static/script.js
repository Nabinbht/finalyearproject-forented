$(document).ready(function () {
    const fileInput = $("#fileInput");
    const previewImg = $("#previewImg");
    const compressedImg = $("#compressedImg");
    const qualitySlider = $("#quality");
    const qualityValue = $("#qualityValue");

    // Handle file selection and preview
    fileInput.on("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.attr("src", e.target.result).show();
            };
            reader.readAsDataURL(file);
        }
    });

    // Update quality value display
    qualitySlider.on("input", function () {
        qualityValue.text($(this).val());
    });

    // Handle form submission
    $("#uploadForm").on("submit", function (e) {
        e.preventDefault();
        var formData = new FormData();
        formData.append("file", fileInput[0].files[0]);
        formData.append("quality", qualitySlider.val()); // Send quality value

        $.ajax({
            url: "/upload",  // Flask route for handling upload
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                console.log("Response:", response);
                if (response.compressed_image_url) {
                    compressedImg.attr("src", response.compressed_image_url).show();
                } else {
                    alert("Compression failed!");
                }
            },
            error: function (xhr, status, error) {
                console.error("Upload failed:", xhr.responseText);
            }
        });
    });
});
