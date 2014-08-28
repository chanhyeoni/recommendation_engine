var products = []
$.ajax({
    type: "get",
    url: "./static/js/products_data.csv",
    async: false,
    dataType:'text',
    success: function(csvd) {
        products = csvd.split("\n");
    }
});